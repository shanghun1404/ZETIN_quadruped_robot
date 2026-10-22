#include "qd_protocol.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static unsigned failed;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: FAIL %s\n", __func__, __LINE__, #x); ++failed; return; } } while (0)

/* Independently calculated CRC, header offsets and float byte order. */
static const uint8_t hello_golden[] = {
    0x51,0x44,0x01,0x01,0x78,0x56,0x34,0x12,0xfe,0xff,0xff,0xff,
    0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0x00,0x42,0x84
};
static qd_frame make_frame(uint8_t type, uint32_t session, uint32_t sequence) {
    qd_frame f = {0}; f.type=type; f.session=session; f.sequence=sequence; return f;
}
static qd_command make_command(uint32_t enabled) {
    qd_command c = {0}; c.enabled = enabled; return c;
}
static qd_frame command_frame(qd_command c, uint32_t session, uint32_t sequence) {
    qd_frame f = make_frame(QD_COMMAND, session, sequence);
    f.payload_length=QD_COMMAND_SIZE;
    if (!qd_pack_command(&c, f.payload)) memset(f.payload, 0, QD_COMMAND_SIZE);
    return f;
}
static void crc_and_frame_vector(void) {
    qd_frame f, decoded;
    uint8_t bytes[QD_MAX_FRAME];
    CHECK(qd_crc16((const uint8_t *)"123456789", 9) == 0x29b1);
    f=make_frame(QD_HELLO, 0x12345678, 0xfffffffe);
    f.stamp_ns=UINT64_C(0x0102030405060708);
    CHECK(qd_encode_frame(&f,bytes,sizeof(bytes))==24);
    CHECK(memcmp(bytes,hello_golden,22)==0);
    CHECK(bytes[22]==0x42 && bytes[23]==0x84);
    CHECK(qd_decode_frame(bytes,24,&decoded));
    CHECK(decoded.session==0x12345678 && decoded.sequence==0xfffffffe);
    CHECK(decoded.stamp_ns==UINT64_C(0x0102030405060708));
    bytes[3]=7; CHECK(!qd_decode_frame(bytes,24,&decoded));
    f.session=0; CHECK(qd_encode_frame(&f,bytes,sizeof(bytes))==0);
    f.session=1; CHECK(qd_encode_frame(&f,bytes,23)==0);
    f.type=QD_STOP; f.payload_length=1; CHECK(qd_encode_frame(&f,bytes,sizeof(bytes))==0);
}
typedef struct { unsigned count; qd_frame frames[8]; } collected;
static void collect(void *user,const qd_frame *frame) {
    collected *c=user;
    if(c->count<8) c->frames[c->count]=*frame;
    ++c->count;
}
static void parser_split_noise_coalesced(void) {
    qd_parser p; collected c={0}; uint8_t bytes[QD_MAX_FRAME], stream[100]; size_t n;
    qd_frame f=make_frame(QD_HELLO,42,10);
    qd_parser_init(&p); n=qd_encode_frame(&f,bytes,sizeof(bytes)); CHECK(n==24);
    for(size_t i=0;i<n-1;++i) qd_parser_feed(&p,bytes+i,1,collect,&c);
    CHECK(c.count==0);
    qd_parser_feed(&p,bytes+n-1,1,collect,&c); CHECK(c.count==1);
    memcpy(stream,"junkQ",5); memcpy(stream+5,bytes,n); memcpy(stream+5+n,bytes,n);
    qd_parser_feed(&p,stream,5+2*n,collect,&c);
    CHECK(c.count==3 && p.used==0);
    CHECK(c.frames[2].sequence==10);
}
static void parser_recovers_corrupt_headers_and_crc(void) {
    qd_parser p; collected c={0}; uint8_t a[QD_MAX_FRAME], b[QD_MAX_FRAME], stream[QD_MAX_FRAME*2];
    qd_frame f=make_frame(QD_HELLO,55,3); size_t n=qd_encode_frame(&f,b,sizeof(b)); CHECK(n==24);
    qd_parser_init(&p); memcpy(a,b,n); a[22]^=0x80;
    memcpy(stream,a,n); memcpy(stream+n,b,n);
    qd_parser_feed(&p,stream,2*n,collect,&c); CHECK(c.count==1);
    memcpy(a,b,n); a[20]=0xff; a[21]=0x7f;
    memcpy(stream,a,n); memcpy(stream+n,b,n);
    qd_parser_feed(&p,stream,2*n,collect,&c); CHECK(c.count==2);
    /* Plausible but incomplete COMMAND must not block a later valid HELLO. */
    memcpy(a,b,22); a[3]=QD_COMMAND; a[20]=0xf4; a[21]=0;
    memcpy(stream,a,22); memcpy(stream+22,b,n);
    qd_parser_feed(&p,stream,22+n,collect,&c); CHECK(c.count==3);
    memcpy(a,b,n); a[2]=2; memcpy(stream,a,n); memcpy(stream+n,b,n);
    qd_parser_feed(&p,stream,2*n,collect,&c); CHECK(c.count==4);
    memset(stream,'Q',sizeof(stream)); qd_parser_feed(&p,stream,sizeof(stream),collect,&c);
    CHECK(p.used<QD_MAX_FRAME);
    qd_parser_feed(&p,b,n,collect,&c); CHECK(c.count==5);
}
static void payload_contract_and_numerics(void) {
    qd_command c=make_command(1), decoded;
    qd_feedback fb={0}, dfb;
    uint8_t bytes[QD_COMMAND_SIZE], fbytes[QD_FEEDBACK_SIZE];
    c.position[0]=1.0f; c.velocity[1]=-2.0f; c.kp[2]=300.0f; c.kd[3]=8.0f; c.effort[4]=48.0f;
    CHECK(qd_pack_command(&c,bytes));
    CHECK(bytes[0]==1 && bytes[1]==0 && bytes[6]==0x80 && bytes[7]==0x3f);
    CHECK(bytes[56]==0 && bytes[57]==0 && bytes[58]==0 && bytes[59]==0xc0);
    CHECK(qd_unpack_command(bytes,sizeof(bytes),&decoded));
    CHECK(decoded.position[0]==1 && decoded.velocity[1]==-2 && decoded.kp[2]==300 && decoded.kd[3]==8 && decoded.effort[4]==48);
    CHECK(!qd_unpack_command(bytes,sizeof(bytes)-1,&decoded));
    c.enabled=2; CHECK(!qd_pack_command(&c,bytes)); c.enabled=1;
    c.position[0]=12.501f; CHECK(!qd_pack_command(&c,bytes)); c.position[0]=12.5f;
    c.velocity[0]=-50.001f; CHECK(!qd_pack_command(&c,bytes)); c.velocity[0]=-50;
    c.kp[0]=-0.01f; CHECK(!qd_pack_command(&c,bytes)); c.kp[0]=300;
    c.kd[0]=8.01f; CHECK(!qd_pack_command(&c,bytes)); c.kd[0]=8;
    c.effort[0]=24.801f; CHECK(!qd_pack_command(&c,bytes)); c.effort[0]=24.8f;
    CHECK(qd_pack_command(&c,bytes));
    c.effort[1]=NAN; CHECK(!qd_pack_command(&c,bytes)); c.effort[1]=INFINITY; CHECK(!qd_pack_command(&c,bytes));
    fb.acknowledged_sequence=0xaabbccdd; fb.fault_flags=9; fb.measured.position[11]=-1.0f;
    CHECK(qd_pack_feedback(&fb,fbytes)); CHECK(fbytes[0]==0xdd && fbytes[3]==0xaa);
    CHECK(qd_unpack_feedback(fbytes,sizeof(fbytes),&dfb));
    CHECK(dfb.acknowledged_sequence==0xaabbccdd && dfb.measured.position[11]==-1);
    fb.armed=2; CHECK(!qd_pack_feedback(&fb,fbytes)); fb.armed=0;
    fb.measured.effort[4]=NAN; CHECK(!qd_pack_feedback(&fb,fbytes));
}
static void sequence_half_range(void) {
    CHECK(qd_sequence_newer(0,0xffffffff)); CHECK(qd_sequence_newer(1,0xfffffffe));
    CHECK(!qd_sequence_newer(2,2)); CHECK(!qd_sequence_newer(1,2));
    CHECK(!qd_sequence_newer(0x80000000,0)); CHECK(qd_sequence_newer(0x7fffffff,0));
}
typedef struct { bool ready,read_ok,apply_ok; unsigned disarms,applies; qd_measurement measured; float torque[12]; } virtual_board;
static bool board_ready(void *u) { return ((virtual_board *)u)->ready; }
static bool board_read(void *u,qd_measurement *m) { virtual_board *b=u; *m=b->measured; return b->read_ok; }
static bool board_apply(void *u,const qd_command *c,const float *torque) { virtual_board *b=u; (void)c; ++b->applies; memcpy(b->torque,torque,sizeof(b->torque)); return b->apply_ok; }
static void board_disarm(void *u) { ++((virtual_board *)u)->disarms; }
static void init_virtual(qd_controller *c,virtual_board *v) {
    qd_board b={v,board_ready,board_read,board_apply,board_disarm};
    memset(v,0,sizeof(*v)); v->ready=v->read_ok=v->apply_ok=true; qd_controller_init(c,&b);
}
static void hello(qd_controller *c,uint32_t session) {
    qd_frame f=make_frame(QD_HELLO,session,77),reply;
    if(qd_controller_on_frame(c,&f,0,&reply)!=QD_REPLY) { ++failed; fprintf(stderr,"HELLO missing ACK\n"); }
}
static void session_stop_default_disabled(void) {
    qd_controller c; qd_frame r,f; virtual_board v; qd_feedback fb;
    qd_controller_init(&c,NULL); CHECK(!c.armed);
    f=command_frame(make_command(1),10,1); CHECK(qd_controller_on_frame(&c,&f,0,&r)==QD_IGNORED);
    f=make_frame(QD_HELLO,10,42); CHECK(qd_controller_on_frame(&c,&f,0,&r)==QD_REPLY);
    CHECK(r.type==QD_HELLO_ACK && r.session==10 && r.sequence==42 && r.payload_length==4 && r.payload[0]==1);
    CHECK(!c.armed && c.session==10 && c.fault_flags==0);
    f=command_frame(make_command(1),10,1); CHECK(qd_controller_on_frame(&c,&f,1,&r)==QD_REJECTED);
    CHECK(!c.armed && (c.fault_flags&QD_FAULT_BOARD));
    CHECK(!qd_controller_feedback(&c,100,&r));
    init_virtual(&c,&v); hello(&c,20);
    f=command_frame(make_command(1),20,1); CHECK(qd_controller_on_frame(&c,&f,1,&r)==QD_ACCEPTED); CHECK(c.armed);
    f=make_frame(QD_STOP,99,2); CHECK(qd_controller_on_frame(&c,&f,2,&r)==QD_IGNORED); CHECK(c.armed);
    f.session=20; CHECK(qd_controller_on_frame(&c,&f,2,&r)==QD_ACCEPTED); CHECK(!c.armed);
    CHECK(qd_controller_feedback(&c,123,&r)); CHECK(qd_unpack_feedback(r.payload,r.payload_length,&fb));
    CHECK(fb.acknowledged_sequence==1 && fb.armed==0 && r.stamp_ns==123);
}
static void replay_wrong_session_watchdog_wrap(void) {
    qd_controller c; virtual_board v; qd_frame f,r;
    init_virtual(&c,&v); hello(&c,20);
    f=command_frame(make_command(1),20,0xfffffffe); CHECK(qd_controller_on_frame(&c,&f,0xfffffff0,&r)==QD_ACCEPTED);
    CHECK(qd_controller_on_frame(&c,&f,0x10,&r)==QD_IGNORED);
    f.session=21; f.sequence=0; CHECK(qd_controller_on_frame(&c,&f,0x20,&r)==QD_IGNORED);
    f.session=20; f.sequence=0xfffffffd; CHECK(qd_controller_on_frame(&c,&f,0x30,&r)==QD_IGNORED);
    qd_controller_tick(&c,0x53); CHECK(c.armed); qd_controller_tick(&c,0x54);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_WATCHDOG && c.last_command_ms==0xfffffff0);
    f.sequence=0; CHECK(qd_controller_on_frame(&c,&f,0x55,&r)==QD_REJECTED); CHECK(!c.armed);
    hello(&c,22); f.session=22; f.sequence=0xffffffff;
    CHECK(qd_controller_on_frame(&c,&f,10,&r)==QD_ACCEPTED);
    f.sequence=0; CHECK(qd_controller_on_frame(&c,&f,11,&r)==QD_ACCEPTED);
    CHECK(c.acknowledged_sequence==0 && c.last_command_ms==11);
    f=make_frame(QD_HELLO,23,0); CHECK(qd_controller_on_frame(&c,&f,12,&r)==QD_REPLY);
    CHECK(!c.armed && !c.have_command && c.fault_flags==0);
}
static void invalid_command_latches_until_hello(void) {
    qd_controller c; virtual_board v; qd_frame f,r; qd_command cmd=make_command(1);
    init_virtual(&c,&v); hello(&c,8);
    f=command_frame(cmd,8,1); CHECK(qd_controller_on_frame(&c,&f,1,&r)==QD_ACCEPTED);
    /* A raw wire NaN bypasses pack validation and must reach the MCU fault path. */
    f.sequence=2; f.payload[4]=0; f.payload[5]=0; f.payload[6]=0xc0; f.payload[7]=0x7f;
    CHECK(qd_controller_on_frame(&c,&f,2,&r)==QD_REJECTED);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_COMMAND && c.acknowledged_sequence==1 && c.last_command_ms==1);
    f=command_frame(cmd,8,3); CHECK(qd_controller_on_frame(&c,&f,3,&r)==QD_REJECTED);
    f=make_frame(QD_STOP,8,4); CHECK(qd_controller_on_frame(&c,&f,4,&r)==QD_ACCEPTED);
    CHECK(c.fault_flags==QD_FAULT_COMMAND);
    hello(&c,9); f=command_frame(cmd,9,1); CHECK(qd_controller_on_frame(&c,&f,5,&r)==QD_ACCEPTED);
    CHECK(c.armed && c.fault_flags==0);
    cmd.enabled=0; f=command_frame(cmd,9,2); CHECK(qd_controller_on_frame(&c,&f,6,&r)==QD_ACCEPTED);
    CHECK(!c.armed); qd_controller_tick(&c,106); CHECK(c.fault_flags==0);
}
static void repeated_hello_cannot_reset_state_or_replay_after_watchdog(void) {
    qd_controller c; virtual_board v; qd_frame f,r;
    init_virtual(&c,&v); hello(&c,123);
    f=command_frame(make_command(1),123,1);
    CHECK(qd_controller_on_frame(&c,&f,10,&r)==QD_ACCEPTED);
    CHECK(qd_controller_feedback(&c,5000000,&r));
    uint32_t tx=c.tx_sequence; unsigned disarms=v.disarms;
    f=make_frame(QD_HELLO,123,88); f.stamp_ns=111;
    CHECK(qd_controller_on_frame(&c,&f,20,&r)==QD_REPLY);
    CHECK(r.type==QD_HELLO_ACK && r.sequence==88 && r.stamp_ns==111);
    CHECK(c.armed && c.have_command && c.acknowledged_sequence==1);
    CHECK(c.last_command_ms==10 && c.tx_sequence==tx && v.disarms==disarms);
    qd_controller_tick(&c,110); CHECK(!c.armed && c.fault_flags==QD_FAULT_WATCHDOG);
    CHECK(qd_controller_on_frame(&c,&f,111,&r)==QD_REPLY);
    CHECK(c.fault_flags==QD_FAULT_WATCHDOG && c.acknowledged_sequence==1 && c.have_command);
    f=command_frame(make_command(1),123,1);
    CHECK(qd_controller_on_frame(&c,&f,112,&r)==QD_IGNORED);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_WATCHDOG && c.last_command_ms==10);
    f.sequence=2; CHECK(qd_controller_on_frame(&c,&f,113,&r)==QD_REJECTED);
    f=make_frame(QD_HELLO,124,0); CHECK(qd_controller_on_frame(&c,&f,114,&r)==QD_REPLY);
    CHECK(c.fault_flags==0 && !c.armed && !c.have_command);
}
static void controller_target_float_rounding_is_bounded_by_gain_and_ulp(void) {
    qd_command c=make_command(1); qd_measurement m={0}; float total[12];
    m.position[2]=-2.0466973781585693f;
    m.velocity[2]=-3.38153338432312f;
    c.position[2]=-2.2845590114593506f; c.kp[2]=300; c.kd[2]=8;
    CHECK(qd_guard_command(&c,&m,total));
    CHECK(fabsf(total[2]-(-44.306222915649414f))<0.00001f);
    c.position[2]=nextafterf(nextafterf(c.position[2],-INFINITY),-INFINITY);
    CHECK(!qd_guard_command(&c,&m,total));
    /* Gain-related slack must still never permit acceleration at overspeed. */
    c=make_command(1); m=(qd_measurement){0}; m.velocity[2]=50;
    c.position[2]=nextafterf(2.0f,INFINITY); m.position[2]=2;
    c.kp[2]=300; CHECK(!qd_guard_command(&c,&m,total));
    /* Near zero, the slack must fit within the remaining 1% curve margin. */
    c=make_command(1); m=(qd_measurement){0}; m.velocity[2]=49.99999f;
    c.effort[2]=0.00002f; CHECK(!qd_guard_command(&c,&m,total));
}
static void total_pd_feedforward_speed_guard(void) {
    qd_command c=make_command(1); qd_measurement m={0}; float total[12];
    c.kp[0]=100; c.position[0]=0.1f; c.kd[0]=1; c.velocity[0]=1; c.effort[0]=2;
    m.position[0]=0.02f; m.velocity[0]=3;
    CHECK(qd_guard_command(&c,&m,total)); CHECK(fabsf(total[0]-8.0f)<0.00001f);
    c.position[0]=0.5f; CHECK(!qd_guard_command(&c,&m,total));
    c=make_command(1); m=(qd_measurement){0}; m.velocity[0]=25;
    c.effort[0]=12.276f; CHECK(qd_guard_command(&c,&m,total));
    c.effort[0]=12.28f; CHECK(!qd_guard_command(&c,&m,total));
    c.effort[0]=-24.5f; CHECK(qd_guard_command(&c,&m,total));
    m.velocity[0]=50; c.effort[0]=0.01f; CHECK(!qd_guard_command(&c,&m,total));
    c.effort[0]=-24.5f; CHECK(qd_guard_command(&c,&m,total));
    m.velocity[0]=-25; c.effort[0]=-12.28f; CHECK(!qd_guard_command(&c,&m,total));
    c.effort[0]=24.5f; CHECK(qd_guard_command(&c,&m,total));
    m.velocity[0]=0; c.effort[0]=24.8f; CHECK(!qd_guard_command(&c,&m,total));
    c.effort[0]=0; c.effort[1]=47.52f; CHECK(qd_guard_command(&c,&m,total));
    /* A float32 PD target at the Python guard boundary reconstructs this
     * total. Rejecting it would stop a correctly guarded command. */
    c.effort[1]=47.5200157f; CHECK(qd_guard_command(&c,&m,total));
    c.effort[1]=47.52005f; CHECK(!qd_guard_command(&c,&m,total));
    c.effort[1]=47.53f; CHECK(!qd_guard_command(&c,&m,total));
    c.effort[1]=0; m.position[0]=NAN; CHECK(!qd_guard_command(&c,&m,total));
}
static void periodic_guard_board_state_failures(void) {
    qd_controller c; virtual_board v; qd_frame f,r; qd_command cmd=make_command(1);
    init_virtual(&c,&v); hello(&c,1); cmd.effort[0]=10;
    f=command_frame(cmd,1,1); CHECK(qd_controller_on_frame(&c,&f,1,&r)==QD_ACCEPTED);
    CHECK(v.applies==1 && v.torque[0]==10);
    v.measured.velocity[0]=49; qd_controller_tick(&c,6);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_GUARD && v.applies==1);
    hello(&c,2); v.measured.velocity[0]=0; v.apply_ok=false;
    f=command_frame(cmd,2,1); CHECK(qd_controller_on_frame(&c,&f,7,&r)==QD_REJECTED);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_BOARD);
    hello(&c,3); v.apply_ok=true; v.read_ok=false;
    f.session=3; CHECK(qd_controller_on_frame(&c,&f,8,&r)==QD_REJECTED);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_STATE);
    hello(&c,4); v.read_ok=true; v.measured.position[0]=INFINITY;
    f.session=4; CHECK(qd_controller_on_frame(&c,&f,9,&r)==QD_REJECTED);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_STATE);
    hello(&c,5); v.measured.position[0]=0; v.ready=false;
    f.session=5; CHECK(qd_controller_on_frame(&c,&f,10,&r)==QD_REJECTED);
    CHECK(!c.armed && c.fault_flags==QD_FAULT_BOARD);
}
int main(void) {
    struct {const char *name; void (*run)(void);} tests[]={
        {"crc_and_frame_vector",crc_and_frame_vector},
        {"parser_split_noise_coalesced",parser_split_noise_coalesced},
        {"parser_recovers_corrupt_headers_and_crc",parser_recovers_corrupt_headers_and_crc},
        {"payload_contract_and_numerics",payload_contract_and_numerics},
        {"sequence_half_range",sequence_half_range},
        {"session_stop_default_disabled",session_stop_default_disabled},
        {"replay_wrong_session_watchdog_wrap",replay_wrong_session_watchdog_wrap},
        {"invalid_command_latches_until_hello",invalid_command_latches_until_hello},
        {"repeated_hello_cannot_reset_state_or_replay_after_watchdog",repeated_hello_cannot_reset_state_or_replay_after_watchdog},
        {"controller_target_float_rounding_is_bounded_by_gain_and_ulp",controller_target_float_rounding_is_bounded_by_gain_and_ulp},
        {"total_pd_feedforward_speed_guard",total_pd_feedforward_speed_guard},
        {"periodic_guard_board_state_failures",periodic_guard_board_state_failures}
    };
    for(size_t i=0;i<sizeof(tests)/sizeof(tests[0]);++i) {unsigned before=failed; tests[i].run(); printf("%s %s\n",failed==before?"PASS":"FAIL",tests[i].name);}
    printf("%u failure(s)\n",failed); return failed?1:0;
}

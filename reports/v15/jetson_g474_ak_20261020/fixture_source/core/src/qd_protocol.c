#include "qd_protocol.h"
#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
               FLT_MAX_EXP == 128, "USB v1 requires IEEE754 binary32");
static uint16_t get16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
}
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t get64(const uint8_t *p) {
    return (uint64_t)get32(p) | ((uint64_t)get32(p+4) << 32);
}
static void put16(uint8_t *p, uint16_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v >> 8);
}
static void put32(uint8_t *p, uint32_t v) {
    for (unsigned i=0; i<4; ++i) p[i]=(uint8_t)(v >> (8*i));
}
static void put64(uint8_t *p, uint64_t v) {
    put32(p,(uint32_t)v); put32(p+4,(uint32_t)(v >> 32));
}
static float get_float(const uint8_t *p) {
    uint32_t bits=get32(p); float value; memcpy(&value,&bits,4); return value;
}
static void put_float(uint8_t *p,float value) {
    uint32_t bits; memcpy(&bits,&value,4); put32(p,bits);
}
static bool shape(uint8_t type,uint16_t length) {
    switch (type) {
        case QD_HELLO: case QD_STOP: return length==0;
        case QD_HELLO_ACK: return length==4;
        case QD_COMMAND: return length==QD_COMMAND_SIZE;
        case QD_FEEDBACK: return length==QD_FEEDBACK_SIZE;
        default: return false;
    }
}
uint16_t qd_crc16(const uint8_t *data,size_t length) {
    uint16_t crc=0xffffu;
    for (size_t i=0; i<length; ++i) {
        crc ^= (uint16_t)((uint16_t)data[i] << 8);
        for (unsigned bit=0; bit<8; ++bit)
            crc=(uint16_t)((crc & 0x8000u) ? ((uint32_t)crc << 1)^0x1021u : (uint32_t)crc << 1);
    }
    return crc;
}
size_t qd_encode_frame(const qd_frame *frame,uint8_t *out,size_t capacity) {
    if (frame==NULL || out==NULL || frame->session==0 ||
        frame->payload_length>QD_MAX_PAYLOAD || !shape(frame->type,frame->payload_length)) return 0;
    size_t n=QD_HEADER_SIZE+frame->payload_length+2u;
    if (capacity<n) return 0;
    out[0]='Q'; out[1]='D'; out[2]=1; out[3]=frame->type;
    put32(out+4,frame->session); put32(out+8,frame->sequence);
    put64(out+12,frame->stamp_ns); put16(out+20,frame->payload_length);
    memcpy(out+QD_HEADER_SIZE,frame->payload,frame->payload_length);
    put16(out+n-2,qd_crc16(out,n-2)); return n;
}
static bool valid_header(const uint8_t *bytes) {
    uint16_t n=get16(bytes+20);
    return bytes[0]=='Q' && bytes[1]=='D' && bytes[2]==1 && get32(bytes+4)!=0 &&
           n<=QD_MAX_PAYLOAD && shape(bytes[3],n);
}
bool qd_decode_frame(const uint8_t *bytes,size_t length,qd_frame *out) {
    if (bytes==NULL || out==NULL || length<QD_HEADER_SIZE+2u || !valid_header(bytes)) return false;
    uint16_t payload_length=get16(bytes+20);
    if (length!=QD_HEADER_SIZE+payload_length+2u ||
        get16(bytes+length-2)!=qd_crc16(bytes,length-2)) return false;
    memset(out,0,sizeof(*out)); out->type=bytes[3]; out->session=get32(bytes+4);
    out->sequence=get32(bytes+8); out->stamp_ns=get64(bytes+12);
    out->payload_length=payload_length;
    memcpy(out->payload,bytes+QD_HEADER_SIZE,payload_length); return true;
}
void qd_parser_init(qd_parser *parser) { memset(parser,0,sizeof(*parser)); }
static void discard(qd_parser *p,size_t count) {
    p->used-=count; memmove(p->bytes,p->bytes+count,p->used);
}
static void parse_available(qd_parser *p,qd_frame_callback callback,void *user) {
    while (p->used>0) {
        size_t start=0;
        while (start+1<p->used && !(p->bytes[start]=='Q' && p->bytes[start+1]=='D')) ++start;
        if (start+1==p->used && p->bytes[start]!='Q') ++start;
        if (start>0) discard(p,start);
        if (p->used<QD_HEADER_SIZE) return;
        if (!valid_header(p->bytes)) { ++p->rejected_frames; discard(p,1); continue; }
        size_t n=QD_HEADER_SIZE+get16(p->bytes+20)+2u;
        qd_frame frame;
        if (p->used>=n) {
            if (qd_decode_frame(p->bytes,n,&frame)) {
                discard(p,n); ++p->emitted_frames;
                if (callback!=NULL) callback(user,&frame);
            } else { ++p->rejected_frames; discard(p,1); }
            continue;
        }
        /* Recover a truncated candidate only to a complete later frame with
         * valid CRC. A damaged length must not block the following HELLO. */
        bool recovered=false;
        for (size_t i=1; i+QD_HEADER_SIZE+2u<=p->used; ++i) {
            if (!valid_header(p->bytes+i)) continue;
            size_t next=QD_HEADER_SIZE+get16(p->bytes+i+20)+2u;
            if (next<=p->used-i && qd_decode_frame(p->bytes+i,next,&frame)) {
                ++p->rejected_frames; discard(p,i); recovered=true; break;
            }
        }
        if (!recovered) return;
    }
}
void qd_parser_feed(qd_parser *parser,const uint8_t *bytes,size_t length,
                    qd_frame_callback callback,void *user) {
    if (parser==NULL || (bytes==NULL && length!=0)) return;
    while (length>0) {
        if (parser->used==sizeof(parser->bytes)) { ++parser->rejected_frames; discard(parser,1); }
        size_t count=sizeof(parser->bytes)-parser->used;
        if (count>length) count=length;
        memcpy(parser->bytes+parser->used,bytes,count);
        parser->used+=count; bytes+=count; length-=count;
        /* Scan incomplete candidates once per received chunk. Scanning for
         * a later recovery frame after every byte made 200Hz commands exceed
         * the G474's foreground budget while CAN and feedback were active. */
        parse_available(parser,callback,user);
    }
}
static bool command_valid(const qd_command *c) {
    if (c==NULL || c->enabled>1) return false;
    for (unsigned i=0; i<QD_JOINT_COUNT; ++i) {
        float cap=(i%3u==0u)?24.8f:48.0f;
        if (!isfinite(c->position[i]) || fabsf(c->position[i])>12.5f ||
            !isfinite(c->velocity[i]) || fabsf(c->velocity[i])>50.0f ||
            !isfinite(c->kp[i]) || c->kp[i]<0.0f || c->kp[i]>300.0f ||
            !isfinite(c->kd[i]) || c->kd[i]<0.0f || c->kd[i]>8.0f ||
            !isfinite(c->effort[i]) || fabsf(c->effort[i])>cap) return false;
    }
    return true;
}
bool qd_pack_command(const qd_command *command,uint8_t out[QD_COMMAND_SIZE]) {
    if (out==NULL || !command_valid(command)) return false;
    put32(out,command->enabled);
    const float *arrays[]={command->position,command->velocity,command->kp,command->kd,command->effort};
    for (unsigned a=0; a<5; ++a)
        for (unsigned i=0; i<QD_JOINT_COUNT; ++i) put_float(out+4+(a*QD_JOINT_COUNT+i)*4,arrays[a][i]);
    return true;
}
bool qd_unpack_command(const uint8_t *payload,size_t length,qd_command *out) {
    if (payload==NULL || out==NULL || length!=QD_COMMAND_SIZE) return false;
    qd_command decoded={0}; decoded.enabled=get32(payload);
    float *arrays[]={decoded.position,decoded.velocity,decoded.kp,decoded.kd,decoded.effort};
    for (unsigned a=0; a<5; ++a)
        for (unsigned i=0; i<QD_JOINT_COUNT; ++i) arrays[a][i]=get_float(payload+4+(a*QD_JOINT_COUNT+i)*4);
    if (!command_valid(&decoded)) return false;
    *out=decoded; return true;
}
static bool feedback_valid(const qd_feedback *f) {
    if (f==NULL || f->armed>1) return false;
    for (unsigned i=0; i<QD_JOINT_COUNT; ++i)
        if (!isfinite(f->measured.position[i]) || !isfinite(f->measured.velocity[i]) ||
            !isfinite(f->measured.effort[i])) return false;
    return true;
}
bool qd_pack_feedback(const qd_feedback *feedback,uint8_t out[QD_FEEDBACK_SIZE]) {
    if (out==NULL || !feedback_valid(feedback)) return false;
    put32(out,feedback->acknowledged_sequence); put32(out+4,feedback->fault_flags); put32(out+8,feedback->armed);
    const float *arrays[]={feedback->measured.position,feedback->measured.velocity,feedback->measured.effort};
    for (unsigned a=0; a<3; ++a)
        for (unsigned i=0; i<QD_JOINT_COUNT; ++i) put_float(out+12+(a*QD_JOINT_COUNT+i)*4,arrays[a][i]);
    return true;
}
bool qd_unpack_feedback(const uint8_t *payload,size_t length,qd_feedback *out) {
    if (payload==NULL || out==NULL || length!=QD_FEEDBACK_SIZE) return false;
    qd_feedback decoded={0}; decoded.acknowledged_sequence=get32(payload);
    decoded.fault_flags=get32(payload+4); decoded.armed=get32(payload+8);
    float *arrays[]={decoded.measured.position,decoded.measured.velocity,decoded.measured.effort};
    for (unsigned a=0; a<3; ++a)
        for (unsigned i=0; i<QD_JOINT_COUNT; ++i) arrays[a][i]=get_float(payload+12+(a*QD_JOINT_COUNT+i)*4);
    if (!feedback_valid(&decoded)) return false;
    *out=decoded; return true;
}
bool qd_sequence_newer(uint32_t candidate,uint32_t previous) {
    uint32_t delta=candidate-previous; return delta!=0 && delta<UINT32_C(0x80000000);
}

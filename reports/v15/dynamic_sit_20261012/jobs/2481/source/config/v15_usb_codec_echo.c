/* Simulator codec oracle using the same C parser as the G474 firmware.
 * This executable does not emulate CAN, motor dynamics or watchdog timing. */
#include "qd_protocol.h"
#include <stdio.h>

static void received(void *user, const qd_frame *input) {
    qd_frame output=*input;
    uint8_t bytes[QD_MAX_FRAME];
    size_t size;
    (void)user;
    if(input->type==QD_COMMAND) {
        qd_command command;
        if(!qd_unpack_command(input->payload,input->payload_length,&command))return;
        if(!qd_pack_command(&command,output.payload))return;
    } else if(input->type==QD_FEEDBACK) {
        qd_feedback feedback;
        if(!qd_unpack_feedback(input->payload,input->payload_length,&feedback))return;
        if(!qd_pack_feedback(&feedback,output.payload))return;
    }
    size=qd_encode_frame(&output,bytes,sizeof bytes);
    if(size && fwrite(bytes,1,size,stdout)!=size)return;
    fflush(stdout);
}

int main(void) {
    qd_parser parser;
    int byte;
    qd_parser_init(&parser);
    while((byte=getchar())!=EOF) {
        uint8_t value=(uint8_t)byte;
        qd_parser_feed(&parser,&value,1,received,NULL);
    }
    return ferror(stdin) ? 1 : 0;
}

#include "main.h"
#include "qd_g474_port.h"
#include <stdio.h>
#include <string.h>

/* Compile and execute the real reference port with host HAL stand-ins.
 * Including the source lets us observe the registered board.ready consumer. */
#define QD_STM32_HAL_EXAMPLE
#include "../examples/g474_hal_port.c"

UART_HandleTypeDef hlpuart1={1}; TIM_HandleTypeDef htim6={6};
static uint32_t mock_ms,mock_primask;
static uint8_t *mock_dma;
static uint16_t mock_dma_at,mock_dma_size;
static qd_frame transmitted[64];
static unsigned transmitted_count,failed;
static bool inject_error_on_tick_read;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: FAIL %s\n",__func__,__LINE__,#x); ++failed; return; } } while (0)
uint32_t HAL_GetTick(void) {
    if (inject_error_on_tick_read) {
        inject_error_on_tick_read=false;
        /* received_frame captures its epoch before it reads the wall clock. */
        HAL_UART_ErrorCallback(&hlpuart1);
    }
    return mock_ms;
}
uint32_t __get_PRIMASK(void) { return mock_primask; }
void __disable_irq(void) { mock_primask=1; }
void __set_PRIMASK(uint32_t state) { mock_primask=state; }
void __DMB(void) { }
HAL_StatusTypeDef HAL_UARTEx_ReceiveToIdle_DMA(UART_HandleTypeDef *uart,uint8_t *data,uint16_t size) {
    (void)uart; mock_dma=data; mock_dma_size=size; return HAL_OK;
}
HAL_StatusTypeDef HAL_TIM_Base_Start_IT(TIM_HandleTypeDef *timer) { (void)timer; return HAL_OK; }
HAL_StatusTypeDef HAL_UART_Transmit_DMA(UART_HandleTypeDef *uart,uint8_t *data,uint16_t size) {
    if (transmitted_count>=64 || !qd_decode_frame(data,size,&transmitted[transmitted_count])) return HAL_ERROR;
    ++transmitted_count; HAL_UART_TxCpltCallback(uart); return HAL_OK;
}
bool qd_g474_board_ready(void) { return true; }
bool qd_g474_board_read(qd_measurement *measured) { memset(measured,0,sizeof(*measured)); return true; }
bool qd_g474_board_apply(const qd_command *command,const float total[QD_JOINT_COUNT]) { (void)command; (void)total; return true; }
void qd_g474_board_disarm(void) { }
void qd_g474_board_disarm_from_isr(void) { }
static void reset_port(void) {
    mock_ms=103; mock_primask=0; mock_dma_at=0; transmitted_count=0;
    inject_error_on_tick_read=false; qd_g474_start();
}
static void receive(const qd_frame *frame) {
    uint8_t bytes[QD_MAX_FRAME]; size_t n=qd_encode_frame(frame,bytes,sizeof(bytes));
    for (size_t i=0;i<n;++i) {
        mock_dma[mock_dma_at++]=bytes[i];
        if (mock_dma_at==mock_dma_size/2u) HAL_UARTEx_RxEventCallback(&hlpuart1,mock_dma_at);
        if (mock_dma_at==mock_dma_size) { HAL_UARTEx_RxEventCallback(&hlpuart1,mock_dma_at); mock_dma_at=0; }
    }
    HAL_UARTEx_RxEventCallback(&hlpuart1,mock_dma_at);
    qd_g474_poll();
}
static void start_session(uint32_t session) {
    qd_frame frame={0}; frame.type=QD_HELLO; frame.session=session; receive(&frame);
}
static void command(uint32_t session) {
    qd_frame frame={0}; qd_command c={0}; c.enabled=1;
    frame.type=QD_COMMAND; frame.session=session; frame.sequence=1; frame.payload_length=QD_COMMAND_SIZE;
    (void)qd_pack_command(&c,frame.payload); receive(&frame);
}
static qd_frame *last_feedback(void) {
    for (unsigned i=transmitted_count;i>0;--i) if (transmitted[i-1].type==QD_FEEDBACK) return &transmitted[i-1];
    return NULL;
}
static void logical_timer_stamp_ignores_foreground_jitter(void) {
    reset_port(); start_session(1);
    for (unsigned i=1;i<=3;++i) {
        mock_ms=103+i*5; HAL_TIM_PeriodElapsedCallback(&htim6);
        mock_ms+=2; qd_g474_poll();
        qd_frame *frame=last_feedback(); CHECK(frame!=NULL);
        CHECK(frame->stamp_ns==(uint64_t)i*5000000u);
    }
}
static void missed_timer_tick_keeps_fault_and_visible_gap(void) {
    reset_port(); start_session(1); command(1);
    mock_ms+=5; HAL_TIM_PeriodElapsedCallback(&htim6);
    mock_ms+=5; HAL_TIM_PeriodElapsedCallback(&htim6); qd_g474_poll();
    qd_frame *frame=last_feedback(); qd_feedback fb;
    CHECK(frame!=NULL && frame->stamp_ns==10000000u);
    CHECK(qd_unpack_feedback(frame->payload,frame->payload_length,&fb));
    CHECK(fb.armed==0 && (fb.fault_flags&QD_FAULT_IO));
}
static void new_isr_fault_during_hello_cannot_be_erased(void) {
    reset_port(); start_session(10); HAL_UART_ErrorCallback(&hlpuart1);
    inject_error_on_tick_read=true; start_session(11);
    mock_ms+=5; HAL_TIM_PeriodElapsedCallback(&htim6); qd_g474_poll();
    qd_frame *frame=last_feedback(); qd_feedback fb;
    CHECK(frame!=NULL && qd_unpack_feedback(frame->payload,frame->payload_length,&fb));
    CHECK(fb.armed==0 && (fb.fault_flags&QD_FAULT_IO));
}
static void repeated_hello_does_not_clear_irq_gate(void) {
    reset_port(); start_session(10); HAL_UART_ErrorCallback(&hlpuart1); start_session(10);
    CHECK(!controller.board.ready(controller.board.user));
}
int main(void) {
    struct { const char *name; void (*run)(void); } tests[]={
        {"logical_timer_stamp_ignores_foreground_jitter",logical_timer_stamp_ignores_foreground_jitter},
        {"missed_timer_tick_keeps_fault_and_visible_gap",missed_timer_tick_keeps_fault_and_visible_gap},
        {"new_isr_fault_during_hello_cannot_be_erased",new_isr_fault_during_hello_cannot_be_erased},
        {"repeated_hello_does_not_clear_irq_gate",repeated_hello_does_not_clear_irq_gate}
    };
    for(size_t i=0;i<sizeof(tests)/sizeof(tests[0]);++i) { unsigned before=failed; tests[i].run(); printf("%s %s\n",failed==before?"PASS":"FAIL",tests[i].name); }
    printf("%u failure(s)\n",failed); return failed?1:0;
}

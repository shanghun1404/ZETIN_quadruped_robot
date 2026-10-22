/* Integration reference, not a generated or flashable CubeMX project.
 * Define QD_STM32_HAL_EXAMPLE and supply main.h, hlpuart1 and htim6.
 * LPUART1: 921600 8N1, PA2 TX / PA3 RX, RX circular DMA, TX normal DMA.
 * TIM6: update IRQ every 5 ms. See README for required IRQ configuration. */
#ifdef QD_STM32_HAL_EXAMPLE
#include "main.h"
#include "qd_g474_port.h"
#include <string.h>

extern UART_HandleTypeDef hlpuart1;
extern TIM_HandleTypeDef htim6;

#define RX_DMA_SIZE 512u
#define RX_QUEUE_SIZE 2048u
#define TX_SLOTS 4u
static uint8_t rx_dma[RX_DMA_SIZE];
static uint8_t rx_queue[RX_QUEUE_SIZE];
static volatile uint16_t rx_head,rx_tail;
static uint16_t dma_previous;
static volatile uint32_t irq_fault,last_valid_command_ms;
static volatile uint32_t irq_fault_epoch;
static volatile uint64_t logical_tick_ns;
static volatile bool watchdog_armed,tx_completed;
static volatile uint8_t ticks_due;
static qd_parser parser;
static qd_controller controller;
static uint8_t tx_data[TX_SLOTS][QD_MAX_FRAME];
static uint16_t tx_length[TX_SLOTS];
static unsigned tx_head,tx_tail,tx_count;
static bool tx_busy;

static void latch_irq_fault(uint32_t flags) {
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    irq_fault |= flags; ++irq_fault_epoch; watchdog_armed=false;
    qd_g474_board_disarm_from_isr();
    __set_PRIMASK(saved);
}
static bool port_ready(void *user) { (void)user; return irq_fault==0 && qd_g474_board_ready(); }
static bool port_read(void *user,qd_measurement *measured) { (void)user; return qd_g474_board_read(measured); }
static bool port_apply(void *user,const qd_command *command,const float *total) {
    (void)user;
    if (irq_fault!=0) return false;
    bool ok=qd_g474_board_apply(command,total);
    if (!ok || irq_fault!=0) { qd_g474_board_disarm(); return false; }
    return true;
}
static void port_disarm(void *user) { (void)user; watchdog_armed=false; qd_g474_board_disarm(); }
static void publish_watchdog(void) {
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    last_valid_command_ms=controller.last_command_ms;
    watchdog_armed=controller.armed && irq_fault==0;
    __set_PRIMASK(saved);
}
static void queue_frame(const qd_frame *frame) {
    if (tx_count==TX_SLOTS) { latch_irq_fault(QD_FAULT_IO); return; }
    size_t n=qd_encode_frame(frame,tx_data[tx_head],QD_MAX_FRAME);
    if (n==0) { latch_irq_fault(QD_FAULT_IO); return; }
    tx_length[tx_head]=(uint16_t)n; tx_head=(tx_head+1u)%TX_SLOTS; ++tx_count;
}
static void received_frame(void *user,const qd_frame *frame) {
    (void)user; qd_frame reply;
    bool starts_session=frame->type==QD_HELLO && frame->payload_length==0 && frame->session!=controller.session;
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    uint32_t previous_fault=irq_fault,previous_epoch=irq_fault_epoch;
    __set_PRIMASK(saved);
    if (previous_fault!=0) qd_controller_fault(&controller,previous_fault);
    enum qd_result result=qd_controller_on_frame(&controller,frame,HAL_GetTick(),&reply);
    saved=__get_PRIMASK(); __disable_irq();
    /* Only a changed-session HELLO can clear a prior IRQ fault. A new ISR
     * event, including a repeated fault bit, changes the epoch and survives. */
    if (result==QD_REPLY && starts_session && irq_fault_epoch==previous_epoch) irq_fault=0;
    uint32_t pending_fault=irq_fault;
    __set_PRIMASK(saved);
    if (pending_fault!=0) qd_controller_fault(&controller,pending_fault);
    if (result==QD_REPLY) queue_frame(&reply);
    publish_watchdog();
}
void qd_g474_start(void) {
    qd_board board={NULL,port_ready,port_read,port_apply,port_disarm};
    qd_controller_init(&controller,&board); qd_parser_init(&parser);
    dma_previous=0; rx_head=rx_tail=0; ticks_due=0; irq_fault=0; irq_fault_epoch=0; logical_tick_ns=0;
    tx_head=tx_tail=tx_count=0; tx_busy=tx_completed=false;
    if (HAL_UARTEx_ReceiveToIdle_DMA(&hlpuart1,rx_dma,RX_DMA_SIZE)!=HAL_OK)
        latch_irq_fault(QD_FAULT_IO);
    /* Keep half-transfer and transfer-complete events enabled in circular
     * DMA; they bound copies even during a continuous 200 Hz stream. */
    if (HAL_TIM_Base_Start_IT(&htim6)!=HAL_OK) latch_irq_fault(QD_FAULT_IO);
}
void qd_g474_poll(void) {
    if (irq_fault!=0) qd_controller_fault(&controller,irq_fault);
    while (rx_tail!=rx_head) {
        uint8_t byte=rx_queue[rx_tail]; rx_tail=(uint16_t)((rx_tail+1u)%RX_QUEUE_SIZE);
        qd_parser_feed(&parser,&byte,1,received_frame,NULL);
    }
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    uint8_t ticks=ticks_due; ticks_due=0;
    uint64_t stamp_ns=logical_tick_ns;
    __set_PRIMASK(saved);
    if (ticks!=0) {
        qd_frame feedback;
        qd_controller_tick(&controller,HAL_GetTick()); publish_watchdog();
        /* ISR-maintained 5 ms grid; foreground execution may be delayed.
         * Missed periods remain visible as a stamp gap and an IO fault. */
        if (qd_controller_feedback(&controller,stamp_ns,&feedback))
            queue_frame(&feedback);
    }
    if (tx_completed) {
        tx_completed=false;
        if (tx_busy) { tx_busy=false; tx_tail=(tx_tail+1u)%TX_SLOTS; --tx_count; }
    }
    if (!tx_busy && tx_count!=0) {
        tx_busy=true;
        if (HAL_UART_Transmit_DMA(&hlpuart1,tx_data[tx_tail],tx_length[tx_tail])!=HAL_OK) {
            tx_busy=false; latch_irq_fault(QD_FAULT_IO);
        }
    }
}
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *uart,uint16_t size) {
    if (uart!=&hlpuart1) return;
    if (size>RX_DMA_SIZE) { latch_irq_fault(QD_FAULT_IO); return; }
    /* HAL reports the DMA write position, including RX_DMA_SIZE at TC. */
    uint16_t end=(uint16_t)(size%RX_DMA_SIZE);
    while (dma_previous!=end) {
        uint16_t next=(uint16_t)((rx_head+1u)%RX_QUEUE_SIZE);
        if (next==rx_tail) { latch_irq_fault(QD_FAULT_IO); break; }
        rx_queue[rx_head]=rx_dma[dma_previous]; __DMB(); rx_head=next;
        dma_previous=(uint16_t)((dma_previous+1u)%RX_DMA_SIZE);
    }
}
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *uart) { if (uart==&hlpuart1) tx_completed=true; }
void HAL_UART_ErrorCallback(UART_HandleTypeDef *uart) {
    if (uart==&hlpuart1) latch_irq_fault(QD_FAULT_IO);
    /* Recover/reinitialize RX in foreground using the selected HAL version.
     * HELLO cannot recover a UART whose DMA has stopped. */
}
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *timer) {
    if (timer!=&htim6) return;
    logical_tick_ns+=UINT64_C(5000000);
    if (ticks_due<255u) ++ticks_due;
    if (ticks_due>1u) latch_irq_fault(QD_FAULT_IO);
    if (watchdog_armed && (uint32_t)(HAL_GetTick()-last_valid_command_ms)>=QD_WATCHDOG_MS)
        latch_irq_fault(QD_FAULT_WATCHDOG);
}
#else
typedef int qd_g474_hal_example_requires_generated_project;
#endif

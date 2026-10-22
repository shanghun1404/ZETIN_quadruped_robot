/* NUCLEO-G474RE firmware for real AK MIT motors.
 * USB: STLINK VCP -> LPUART1 PA2/PA3, 921600 8N1, qd_protocol.
 * CAN: FDCAN1 PA12/PA11, FDCAN2 PB13/PB12, FDCAN3 PA15/PA8, 1 Mbit/s normal mode
 * (user wiring, bench-verified with TJA1051T/3 on 2026-10-20).
 * Motors are driven only through qd_ak_motor with a verified g474_ak_config.
 * There is no hardware power cut: if this MCU or the CAN link stops, AK motors
 * keep their last command. A separate emergency power cut is required. */
#include "stm32g4xx_hal.h"
#include "qd_protocol.h"
#include "qd_ak_motor.h"
#include "g474_ak_diag.h"
#include <string.h>

#define RX_SIZE 4096u
#define CAN_RING 64u
#define TX_WAIT_CYCLES 80000u /* 0.5 ms at 160 MHz, several 1 Mbit/s frames */

extern const qd_ak_motor_config g474_ak_config;
volatile g474_ak_diag g_diag __attribute__((section(".diag")));
static UART_HandleTypeDef uart;
static FDCAN_HandleTypeDef can[3];
static uint8_t rx_byte,rx_ring[RX_SIZE];
static volatile uint32_t rx_head,rx_tail;
static qd_ak_can_frame can_ring[CAN_RING];
static volatile uint32_t can_head,can_tail;
static bool bus_off_active[3];
static qd_controller controller;
static qd_parser parser;
static qd_ak_motor motor;

static void fatal(void) { g_diag.boot_stage |= 0x80000000u; __disable_irq(); while(1) __WFI(); }
void HardFault_Handler(void) { fatal(); }
void MemManage_Handler(void) { fatal(); }
void BusFault_Handler(void) { fatal(); }
void UsageFault_Handler(void) { fatal(); }
void SysTick_Handler(void) { HAL_IncTick(); }
void LPUART1_IRQHandler(void) { HAL_UART_IRQHandler(&uart); }
void FDCAN1_IT0_IRQHandler(void) { HAL_FDCAN_IRQHandler(&can[0]); }
void FDCAN2_IT0_IRQHandler(void) { HAL_FDCAN_IRQHandler(&can[1]); }
void FDCAN3_IT0_IRQHandler(void) { HAL_FDCAN_IRQHandler(&can[2]); }
void HAL_MspInit(void) { __HAL_RCC_PWR_CLK_ENABLE(); HAL_PWREx_DisableUCPDDeadBattery(); }

static void clock_init(void) {
 if(HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST)!=HAL_OK) fatal();
 RCC_OscInitTypeDef o={0}; o.OscillatorType=RCC_OSCILLATORTYPE_HSI; o.HSIState=RCC_HSI_ON;
 o.HSICalibrationValue=RCC_HSICALIBRATION_DEFAULT; o.PLL.PLLState=RCC_PLL_ON;
 o.PLL.PLLSource=RCC_PLLSOURCE_HSI; o.PLL.PLLM=RCC_PLLM_DIV4; o.PLL.PLLN=80;
 o.PLL.PLLP=RCC_PLLP_DIV2; o.PLL.PLLQ=RCC_PLLQ_DIV2; o.PLL.PLLR=RCC_PLLR_DIV2;
 if(HAL_RCC_OscConfig(&o)!=HAL_OK) fatal();
 RCC_ClkInitTypeDef c={0}; c.ClockType=RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
 c.SYSCLKSource=RCC_SYSCLKSOURCE_PLLCLK; c.AHBCLKDivider=RCC_SYSCLK_DIV1;
 c.APB1CLKDivider=RCC_HCLK_DIV1; c.APB2CLKDivider=RCC_HCLK_DIV1;
 if(HAL_RCC_ClockConfig(&c,FLASH_LATENCY_4)!=HAL_OK) fatal();
 RCC_PeriphCLKInitTypeDef p={0}; p.PeriphClockSelection=RCC_PERIPHCLK_LPUART1|RCC_PERIPHCLK_FDCAN;
 p.Lpuart1ClockSelection=RCC_LPUART1CLKSOURCE_PCLK1; p.FdcanClockSelection=RCC_FDCANCLKSOURCE_PLL;
 if(HAL_RCCEx_PeriphCLKConfig(&p)!=HAL_OK) fatal();
 g_diag.clock_hz=HAL_RCC_GetHCLKFreq();
}
static void uart_init(void) {
 __HAL_RCC_GPIOA_CLK_ENABLE(); __HAL_RCC_LPUART1_CLK_ENABLE();
 GPIO_InitTypeDef p={0}; p.Pin=GPIO_PIN_2|GPIO_PIN_3; p.Mode=GPIO_MODE_AF_PP;
 p.Pull=GPIO_NOPULL; p.Speed=GPIO_SPEED_FREQ_VERY_HIGH; p.Alternate=GPIO_AF12_LPUART1;
 HAL_GPIO_Init(GPIOA,&p);
 uart.Instance=LPUART1; uart.Init.BaudRate=921600; uart.Init.WordLength=UART_WORDLENGTH_8B;
 uart.Init.StopBits=UART_STOPBITS_1; uart.Init.Parity=UART_PARITY_NONE; uart.Init.Mode=UART_MODE_TX_RX;
 uart.Init.HwFlowCtl=UART_HWCONTROL_NONE; uart.Init.OverSampling=UART_OVERSAMPLING_16;
 uart.Init.OneBitSampling=UART_ONE_BIT_SAMPLE_DISABLE; uart.Init.ClockPrescaler=UART_PRESCALER_DIV1;
 if(HAL_UART_Init(&uart)!=HAL_OK) fatal();
 HAL_NVIC_SetPriority(LPUART1_IRQn,1,0); HAL_NVIC_EnableIRQ(LPUART1_IRQn);
 if(HAL_UART_Receive_IT(&uart,&rx_byte,1)!=HAL_OK) fatal();
}
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *h) {
 if(h!=&uart) return;
 g_diag.rx_bytes++; uint32_t next=(rx_head+1u)&(RX_SIZE-1u);
 if(next==rx_tail) g_diag.rx_overflows++;
 else { rx_ring[rx_head]=rx_byte; __DMB(); rx_head=next; }
 if(HAL_UART_Receive_IT(&uart,&rx_byte,1)!=HAL_OK) g_diag.uart_errors++;
}
void HAL_UART_ErrorCallback(UART_HandleTypeDef *h) {
 if(h!=&uart) return;
 g_diag.uart_errors++; (void)HAL_UART_Receive_IT(&uart,&rx_byte,1);
}

static unsigned bus_of(const FDCAN_HandleTypeDef *h) { return h==&can[0]?0u:(h==&can[1]?1u:2u); }
/* ISR producer; all three FDCAN IRQs share one priority, so one writer. */
static void drain_fifo(FDCAN_HandleTypeDef *h,uint32_t fifo) {
 unsigned b=bus_of(h);
 while(HAL_FDCAN_GetRxFifoFillLevel(h,fifo)!=0u) {
  FDCAN_RxHeaderTypeDef r; uint8_t data[8]={0};
  if(HAL_FDCAN_GetRxMessage(h,fifo,&r,data)!=HAL_OK) { g_diag.bus[b].rx_lost++; return; }
  g_diag.bus[b].rx_frames++;
  if(r.RxFrameType!=FDCAN_DATA_FRAME||r.FDFormat!=FDCAN_CLASSIC_CAN||r.DataLength>FDCAN_DLC_BYTES_8) continue;
  uint32_t next=(can_head+1u)&(CAN_RING-1u);
  if(next==can_tail) { g_diag.can_ring_overflows++; continue; }
  qd_ak_can_frame *f=&can_ring[can_head];
  f->bus=b+1u; f->extended=r.IdType==FDCAN_EXTENDED_ID; f->id=r.Identifier; f->dlc=r.DataLength;
  memcpy(f->data,data,8); __DMB(); can_head=next;
 }
}
void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *h,uint32_t its) {
 if(its&FDCAN_IT_RX_FIFO0_MESSAGE_LOST) g_diag.bus[bus_of(h)].rx_lost++;
 drain_fifo(h,FDCAN_RX_FIFO0);
}
void HAL_FDCAN_RxFifo1Callback(FDCAN_HandleTypeDef *h,uint32_t its) {
 if(its&FDCAN_IT_RX_FIFO1_MESSAGE_LOST) g_diag.bus[bus_of(h)].rx_lost++;
 drain_fifo(h,FDCAN_RX_FIFO1);
}
static void can_init(void) {
 __HAL_RCC_GPIOA_CLK_ENABLE(); __HAL_RCC_GPIOB_CLK_ENABLE(); __HAL_RCC_FDCAN_CLK_ENABLE();
 /* PA15 uses AF11 instead of JTAG TDI. SWD retains PA13/PA14. */
 CLEAR_BIT(DBGMCU->CR,DBGMCU_CR_TRACE_IOEN);
 GPIO_InitTypeDef p={0}; p.Mode=GPIO_MODE_AF_PP; p.Pull=GPIO_PULLUP; p.Speed=GPIO_SPEED_FREQ_VERY_HIGH;
 /* User schematic: CN10-12/14, CN10-30/16, CN7-17/CN10-23. */
 p.Pin=GPIO_PIN_11|GPIO_PIN_12; p.Alternate=GPIO_AF9_FDCAN1; HAL_GPIO_Init(GPIOA,&p);
 p.Pin=GPIO_PIN_12|GPIO_PIN_13; p.Alternate=GPIO_AF9_FDCAN2; HAL_GPIO_Init(GPIOB,&p);
 p.Pin=GPIO_PIN_8|GPIO_PIN_15; p.Alternate=GPIO_AF11_FDCAN3; HAL_GPIO_Init(GPIOA,&p);
 for(unsigned i=0;i<3;i++) {
  can[i].Instance=i==0?FDCAN1:(i==1?FDCAN2:FDCAN3);
  FDCAN_InitTypeDef *c=&can[i].Init;
  c->ClockDivider=FDCAN_CLOCK_DIV2; c->FrameFormat=FDCAN_FRAME_CLASSIC; c->Mode=FDCAN_MODE_NORMAL;
  /* One attempt per frame: a missing ACK must not flood the bus. */
  c->AutoRetransmission=DISABLE; c->TransmitPause=DISABLE; c->ProtocolException=DISABLE;
  c->NominalPrescaler=5; c->NominalSyncJumpWidth=2; c->NominalTimeSeg1=13; c->NominalTimeSeg2=2;
  c->DataPrescaler=5; c->DataSyncJumpWidth=2; c->DataTimeSeg1=13; c->DataTimeSeg2=2;
  c->StdFiltersNbr=0; c->ExtFiltersNbr=0; c->TxFifoQueueMode=FDCAN_TX_FIFO_OPERATION;
  if(HAL_FDCAN_Init(&can[i])!=HAL_OK) fatal();
  /* Standard replies (LEGACY) -> FIFO0, extended (V3) -> FIFO1. */
  if(HAL_FDCAN_ConfigGlobalFilter(&can[i],FDCAN_ACCEPT_IN_RX_FIFO0,FDCAN_ACCEPT_IN_RX_FIFO1,FDCAN_REJECT_REMOTE,FDCAN_REJECT_REMOTE)!=HAL_OK) fatal();
  if(HAL_FDCAN_ActivateNotification(&can[i],FDCAN_IT_RX_FIFO0_NEW_MESSAGE|FDCAN_IT_RX_FIFO0_MESSAGE_LOST|
     FDCAN_IT_RX_FIFO1_NEW_MESSAGE|FDCAN_IT_RX_FIFO1_MESSAGE_LOST,0)!=HAL_OK) fatal();
  if(HAL_FDCAN_Start(&can[i])!=HAL_OK) fatal();
 }
 /* Below LPUART1: a byte arrives every 10.9 us at 921600 baud. */
 HAL_NVIC_SetPriority(FDCAN1_IT0_IRQn,2,0); HAL_NVIC_EnableIRQ(FDCAN1_IT0_IRQn);
 HAL_NVIC_SetPriority(FDCAN2_IT0_IRQn,2,0); HAL_NVIC_EnableIRQ(FDCAN2_IT0_IRQn);
 HAL_NVIC_SetPriority(FDCAN3_IT0_IRQn,2,0); HAL_NVIC_EnableIRQ(FDCAN3_IT0_IRQn);
}
/* Queued, not delivered: the adapter treats a missing reply as stale state. */
static bool can_send(void *user,const qd_ak_can_frame *f) {
 (void)user;
 if(f->bus<1u||f->bus>3u||f->dlc>8u) return false;
 unsigned b=f->bus-1u; FDCAN_HandleTypeDef *h=&can[b];
 FDCAN_TxHeaderTypeDef t={0};
 t.Identifier=f->id; t.IdType=f->extended?FDCAN_EXTENDED_ID:FDCAN_STANDARD_ID;
 t.TxFrameType=FDCAN_DATA_FRAME; t.DataLength=f->dlc; t.ErrorStateIndicator=FDCAN_ESI_ACTIVE;
 t.BitRateSwitch=FDCAN_BRS_OFF; t.FDFormat=FDCAN_CLASSIC_CAN; t.TxEventFifoControl=FDCAN_NO_TX_EVENTS;
 uint32_t begin=DWT->CYCCNT;
 while(HAL_FDCAN_GetTxFifoFreeLevel(h)==0u)
  if((uint32_t)(DWT->CYCCNT-begin)>TX_WAIT_CYCLES) { g_diag.bus[b].tx_full++; return false; }
 if(HAL_FDCAN_AddMessageToTxFifoQ(h,&t,f->data)!=HAL_OK) { g_diag.bus[b].tx_errors++; return false; }
 g_diag.bus[b].tx_queued++; return true;
}
static void process_can(uint32_t now) {
 while(can_tail!=can_head) {
  __DMB(); qd_ak_can_frame f=can_ring[can_tail]; __DMB(); can_tail=(can_tail+1u)&(CAN_RING-1u);
  if(qd_ak_motor_receive(&motor,&f)) continue;
  volatile g474_can_log *e=&g_diag.unknown[g_diag.unknown_count%G474_AK_UNKNOWN_LOG];
  e->bus=f.bus; e->extended=f.extended; e->id=f.id; e->dlc=f.dlc;
  e->data_lo=(uint32_t)f.data[0]|(uint32_t)f.data[1]<<8|(uint32_t)f.data[2]<<16|(uint32_t)f.data[3]<<24;
  e->data_hi=(uint32_t)f.data[4]|(uint32_t)f.data[5]<<8|(uint32_t)f.data[6]<<16|(uint32_t)f.data[7]<<24;
  e->tick_ms=now; g_diag.unknown_count++;
 }
}
/* PSR.LEC resets to "no change" on read, so keep the last reported error. */
static void check_buses(void) {
 for(unsigned b=0;b<3;b++) {
  FDCAN_GlobalTypeDef *r=can[b].Instance; uint32_t psr=r->PSR;
  if((psr&FDCAN_PSR_LEC)!=FDCAN_PSR_LEC) { g_diag.bus[b].last_psr=psr; g_diag.bus[b].last_ecr=r->ECR; }
  bool off=(psr&FDCAN_PSR_BO)!=0u;
  if(off&&!bus_off_active[b]) { g_diag.bus[b].bus_off++; qd_controller_fault(&controller,QD_FAULT_IO); }
  if(off) CLEAR_BIT(r->CCCR,FDCAN_CCCR_INIT); /* bus-off recovery */
  bus_off_active[b]=off;
 }
}
/* One rest frame to an arbitrary ID for reply discovery; never while armed. */
static void run_probe(void) {
 uint32_t request=g_diag.probe_request;
 if(request==g_diag.probe_done) return;
 g_diag.probe_done=request;
 uint32_t bus=g_diag.probe_bus,protocol=g_diag.probe_protocol,id=g_diag.probe_id;
 if(controller.armed||bus<1u||bus>3u||id<1u||id>255u) return;
 qd_ak_can_frame f={0}; f.bus=bus; f.dlc=8;
 if(protocol==QD_AK_LEGACY_MIT) { f.id=id; memset(f.data,0xff,7); f.data[7]=0xfd; }
 else if(protocol==QD_AK_V3_MIT) {
  /* kp=kd=0 with mid-scale position, velocity and torque codes. */
  static const uint8_t rest[8]={0,0,0,0x80,0,0x80,0x08,0};
  f.extended=1; f.id=8u<<8|id; memcpy(f.data,rest,8);
 } else return;
 if(can_send(NULL,&f)) g_diag.probe_sent++;
}
static void send_frame(const qd_frame *f) {
 uint8_t bytes[QD_MAX_FRAME]; size_t n=qd_encode_frame(f,bytes,sizeof(bytes));
 if(!n||HAL_UART_Transmit(&uart,bytes,(uint16_t)n,20)!=HAL_OK) {g_diag.uart_tx_errors++;qd_controller_fault(&controller,QD_FAULT_IO);}
}
static void handle_frame(void *u,const qd_frame *f) {
 (void)u; qd_frame reply;
 enum qd_result status=qd_controller_on_frame(&controller,f,HAL_GetTick(),&reply);
 g_diag.frames_received++;
 if(f->type==QD_COMMAND&&f->session==controller.session) {
  g_diag.commands_received++;
  if(status==QD_ACCEPTED) g_diag.commands_accepted++;
  else if(status==QD_REJECTED) g_diag.commands_rejected++;
 }
 if(status==QD_REPLY) send_frame(&reply);
}
int main(void) {
 memset((void *)&g_diag,0,sizeof(g_diag)); g_diag.magic=G474_AK_DIAG_MAGIC; g_diag.version=G474_AK_DIAG_VERSION;
 g_diag.boot_stage=1;
 HAL_Init(); clock_init(); g_diag.boot_stage=2;
 CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT=0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
 qd_ak_motor_init(&motor,&g474_ak_config,can_send,NULL); g_diag.config_valid=motor.config_valid;
 can_init();
 qd_board board=qd_ak_motor_board(&motor);
 qd_controller_init(&controller,&board); qd_parser_init(&parser);
 uart_init(); g_diag.boot_stage=3;
 uint32_t next=HAL_GetTick()+5u; uint64_t logical=0;
 while(1) {
  uint32_t now=HAL_GetTick();
  qd_ak_motor_clock(&motor,now);
  process_can(now);
  check_buses();
  if(g_diag.rx_overflows||g_diag.uart_errors||g_diag.uart_tx_errors) qd_controller_fault(&controller,QD_FAULT_IO);
  uint8_t bytes[64]; size_t n=0;
  while(rx_tail!=rx_head&&n<sizeof(bytes)) {__DMB();bytes[n++]=rx_ring[rx_tail];__DMB();rx_tail=(rx_tail+1u)&(RX_SIZE-1u);}
  if(n) qd_parser_feed(&parser,bytes,n,handle_frame,NULL);
  now=HAL_GetTick();
  if((int32_t)(now-next)>=0) {
   uint32_t ticks=1u+(now-next)/5u; g_diag.missed_ticks+=ticks-1u; next+=ticks*5u; logical+=(uint64_t)ticks*5000000u;
   /* Re-apply at 200 Hz while armed; otherwise rest-poll for fresh state. */
   qd_controller_tick(&controller,now);
   if(!controller.armed) { if(qd_ak_motor_poll(&motor)) g_diag.polls++; run_probe(); }
   qd_frame f; if(qd_controller_feedback(&controller,logical,&f)) {send_frame(&f);g_diag.feedback_sent++;}
  }
  g_diag.armed=controller.armed; g_diag.fault_flags=controller.fault_flags;
  g_diag.adapter_rejected=motor.rejected_frames; g_diag.adapter_send_failures=motor.send_failures;
 }
}

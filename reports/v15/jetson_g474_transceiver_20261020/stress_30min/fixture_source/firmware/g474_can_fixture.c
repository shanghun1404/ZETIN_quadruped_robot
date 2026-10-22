/* Bare NUCLEO-G474RE diagnostic fixture. No motor-ready/apply path.
 * Disabled QD commands only -> diagnostic CAN frames, never motor enable.
 * Loopback feedback and q/dq/tau below are SYNTHETIC, not motor feedback.
 * GPIOA/B IDR captures physical pads connected to user's TJA1051T/3 board.
 * Capture pins: low16=GPIOA IDR, high16=GPIOB IDR. No external ACK emulation.
 */
#include "stm32g4xx_hal.h"
#include "qd_protocol.h"
#include "qd_ak_virtual.h"
#include "g474_diag.h"
#include <string.h>

#define RX_SIZE 4096u
volatile can_diag g_diag __attribute__((section(".diag")));
volatile pin_sample g_capture[CAPTURE_ROUNDS][CAPTURE_SAMPLES] __attribute__((section(".capture")));
static UART_HandleTypeDef uart;
static FDCAN_HandleTypeDef can[3];
static uint8_t rx_byte,rx_ring[RX_SIZE];
static volatile uint32_t rx_head,rx_tail;
static qd_controller controller;
static qd_parser parser;
static qd_measurement virtual_state;
static uint32_t callback_cycles;

static void fatal(void) { g_diag.boot_stage |= 0x80000000u; __disable_irq(); while(1) __WFI(); }
void HardFault_Handler(void) { fatal(); }
void MemManage_Handler(void) { fatal(); }
void BusFault_Handler(void) { fatal(); }
void UsageFault_Handler(void) { fatal(); }
void SysTick_Handler(void) { HAL_IncTick(); }
void LPUART1_IRQHandler(void) { HAL_UART_IRQHandler(&uart); }
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
static void can_init(uint32_t mode) {
 __HAL_RCC_GPIOA_CLK_ENABLE(); __HAL_RCC_GPIOB_CLK_ENABLE(); __HAL_RCC_FDCAN_CLK_ENABLE();
 /* PA15 uses AF11 instead of JTAG TDI. SWD retains PA13/PA14. */
 CLEAR_BIT(DBGMCU->CR,DBGMCU_CR_TRACE_IOEN);
 GPIO_InitTypeDef p={0}; p.Mode=GPIO_MODE_AF_PP; p.Pull=GPIO_PULLUP; p.Speed=GPIO_SPEED_FREQ_VERY_HIGH;
 /* User schematic: CN10-12/14, CN10-30/16, CN7-17/CN10-23. */
 p.Pin=GPIO_PIN_11|GPIO_PIN_12; p.Alternate=GPIO_AF9_FDCAN1; HAL_GPIO_Init(GPIOA,&p);
 p.Pin=GPIO_PIN_12|GPIO_PIN_13; p.Alternate=GPIO_AF9_FDCAN2; HAL_GPIO_Init(GPIOB,&p);
 p.Pin=GPIO_PIN_8|GPIO_PIN_15; p.Alternate=GPIO_AF11_FDCAN3; HAL_GPIO_Init(GPIOA,&p);
 __HAL_RCC_FDCAN_FORCE_RESET(); __HAL_RCC_FDCAN_RELEASE_RESET();
 memset(can,0,sizeof(can));
 for(unsigned i=0;i<3;i++) {
  can[i].Instance=i==0?FDCAN1:(i==1?FDCAN2:FDCAN3);
  FDCAN_InitTypeDef *c=&can[i].Init;
  c->ClockDivider=FDCAN_CLOCK_DIV2; c->FrameFormat=FDCAN_FRAME_CLASSIC;
  c->Mode=mode==0?FDCAN_MODE_INTERNAL_LOOPBACK:(mode==1?FDCAN_MODE_EXTERNAL_LOOPBACK:FDCAN_MODE_NORMAL);
  c->AutoRetransmission=DISABLE; c->TransmitPause=DISABLE; c->ProtocolException=DISABLE;
  c->NominalPrescaler=5; c->NominalSyncJumpWidth=2; c->NominalTimeSeg1=13; c->NominalTimeSeg2=2;
  c->DataPrescaler=5; c->DataSyncJumpWidth=2; c->DataTimeSeg1=13; c->DataTimeSeg2=2;
  c->StdFiltersNbr=0; c->ExtFiltersNbr=0; c->TxFifoQueueMode=FDCAN_TX_FIFO_OPERATION;
  if(HAL_FDCAN_Init(&can[i])!=HAL_OK) fatal();
  if(HAL_FDCAN_ConfigGlobalFilter(&can[i],FDCAN_ACCEPT_IN_RX_FIFO0,FDCAN_ACCEPT_IN_RX_FIFO0,FDCAN_REJECT_REMOTE,FDCAN_REJECT_REMOTE)!=HAL_OK) fatal();
  if(HAL_FDCAN_Start(&can[i])!=HAL_OK) fatal();
  g_diag.bus[i].cccr=can[i].Instance->CCCR; g_diag.bus[i].nbtp=can[i].Instance->NBTP;
  g_diag.bus[i].test_reg=can[i].Instance->TEST;
 }
 g_diag.current_mode=mode; g_diag.mode_changes++;
 g_diag.gpio_moder=GPIOB->MODER; g_diag.gpio_afrl=GPIOB->AFR[0]; g_diag.gpio_afrh=GPIOB->AFR[1];
 g_diag.gpioa_moder=GPIOA->MODER; g_diag.gpioa_afrl=GPIOA->AFR[0];g_diag.gpioa_afrh=GPIOA->AFR[1];
 g_diag.pwr_cr3=PWR->CR3;
}
static bool never_ready(void *u) { (void)u; g_diag.ready_checks++; return false; }
static bool synthetic_read(void *u,qd_measurement *m) { (void)u; *m=virtual_state; return true; }
static bool never_apply(void *u,const qd_command *c,const float *t) { (void)u;(void)c;(void)t;g_diag.apply_calls++;return false; }
static void disarm(void *u) { (void)u; }
static uint32_t word(const uint8_t *p) { return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }

/* Same message-RAM format/72byte stride as official STM32G4 HAL.
 * Prepare only, so pad recording can start BEFORE the TXBAR trigger. */
static uint32_t prepare_can(unsigned bus,const qd_ak_frame *f) {
 FDCAN_GlobalTypeDef *r=can[bus].Instance;
 if((r->TXFQS&FDCAN_TXFQS_TFQF)!=0u) { g_diag.can_errors++; return 0; }
 uint32_t index=(r->TXFQS&FDCAN_TXFQS_TFQPI)>>FDCAN_TXFQS_TFQPI_Pos;
 volatile uint32_t *ram=(volatile uint32_t *)(can[bus].msgRam.TxFIFOQSA+index*72u);
 ram[0]=f->extended?(f->arbitration_id|FDCAN_EXTENDED_ID):(f->arbitration_id<<18);
 ram[1]=f->dlc<<16; ram[2]=word(f->data); ram[3]=word(f->data+4);
 __DMB(); g_diag.bus[bus].queued++; g_diag.joint[f->joint_index].queued++;
 g_diag.joint[f->joint_index].last_id=f->arbitration_id;
 g_diag.joint[f->joint_index].expected_lo=ram[2]; g_diag.joint[f->joint_index].expected_hi=ram[3];
 return 1u<<index;
}
static void trigger(const uint32_t masks[3]) {
 can[0].Instance->TXBAR=masks[0]; can[1].Instance->TXBAR=masks[1]; can[2].Instance->TXBAR=masks[2];
 __DSB();
}
static void capture(unsigned round,const uint32_t masks[3]) {
 uint32_t saved=__get_PRIMASK(); __disable_irq();
 uint32_t base=DWT->CYCCNT;
 for(unsigned k=0;k<CAPTURE_SAMPLES;k++) {
  while((uint32_t)(DWT->CYCCNT-base)<k*40u) {}
  if(k==24u) trigger(masks); /* 6us idle prefix on all three physical TX pins. */
  uint32_t now=DWT->CYCCNT-base;
  uint32_t pins_a=GPIOA->IDR,pins_b=GPIOB->IDR;
  g_capture[round][k].cycle=now; g_capture[round][k].pins=(pins_a&0xffffu)|(pins_b<<16);
 }
 __set_PRIMASK(saved);
}
static bool finish_can(unsigned bus,const qd_ak_frame *f,uint32_t mask,qd_ak_frame *received) {
 FDCAN_GlobalTypeDef *r=can[bus].Instance;
 uint32_t begin=DWT->CYCCNT;
 while((r->TXBRP&mask)!=0u && (uint32_t)(DWT->CYCCNT-begin)<160000u) {}
 g_diag.bus[bus].txbrp=r->TXBRP; g_diag.bus[bus].txbto=r->TXBTO; g_diag.bus[bus].txbcf=r->TXBCF;
 g_diag.bus[bus].last_psr=r->PSR; g_diag.bus[bus].last_ecr=r->ECR;
 bool ok=(r->TXBTO&mask)!=0u;
 if(ok) { g_diag.bus[bus].completed++; g_diag.joint[f->joint_index].completed++; }
 else { g_diag.bus[bus].tx_cancelled++; g_diag.can_errors++; }
 if(g_diag.current_mode==2u) return ok;
 begin=DWT->CYCCNT;
 while(HAL_FDCAN_GetRxFifoFillLevel(&can[bus],FDCAN_RX_FIFO0)==0u && (uint32_t)(DWT->CYCCNT-begin)<160000u) {}
 if(HAL_FDCAN_GetRxFifoFillLevel(&can[bus],FDCAN_RX_FIFO0)==0u) { g_diag.bus[bus].timeouts++;g_diag.can_errors++;return false; }
 FDCAN_RxHeaderTypeDef h={0}; uint8_t bytes[64]={0};
 if(HAL_FDCAN_GetRxMessage(&can[bus],FDCAN_RX_FIFO0,&h,bytes)!=HAL_OK) { g_diag.can_errors++;return false; }
 g_diag.bus[bus].received++; g_diag.joint[f->joint_index].received++;
 g_diag.joint[f->joint_index].received_lo=word(bytes); g_diag.joint[f->joint_index].received_hi=word(bytes+4);
 if(h.IdType!=(f->extended?FDCAN_EXTENDED_ID:FDCAN_STANDARD_ID) || h.RxFrameType!=FDCAN_DATA_FRAME || h.DataLength!=f->dlc ||
    h.Identifier!=f->arbitration_id || memcmp(bytes,f->data,f->dlc)!=0) {
  g_diag.bus[bus].rx_mismatch++;g_diag.can_errors++;return false;
 }
 if(received) { *received=*f;received->arbitration_id=h.Identifier;received->extended=h.IdType==FDCAN_EXTENDED_ID;
  received->dlc=h.DataLength;memcpy(received->data,bytes,8); }
 return ok;
}
static void route(const qd_command *command,uint32_t sequence) {
 qd_ak_frame frames[12];
 for(unsigned j=0;j<12;j++) if(!qd_ak_build_command(g_diag.current_profile,command,j,&frames[j])) {
  g_diag.route_rejected++;qd_controller_fault(&controller,QD_FAULT_COMMAND);return;
 }
 uint32_t begin=DWT->CYCCNT;
 bool recording=g_diag.capture_requested!=0u;
 unsigned captured_rounds=0;
 bool ok=true;
 for(unsigned leg=0;leg<4;leg++) {
  uint32_t masks[3];qd_ak_frame received[3],replies[3];
  for(unsigned bus=0;bus<3;bus++) masks[bus]=prepare_can(bus,&frames[leg*3+bus]);
  if(!masks[0]||!masks[1]||!masks[2]) { ok=false;break; }
  if(recording) {capture(leg*2u,masks);captured_rounds++;} else trigger(masks);
  for(unsigned bus=0;bus<3;bus++) {
   if(!finish_can(bus,&frames[leg*3+bus],masks[bus],&received[bus])) ok=false;
   if(g_diag.current_mode!=2u && ok && !qd_ak_virtual_reply(g_diag.current_profile,&received[bus],&replies[bus])) {g_diag.virtual_errors++;ok=false;}
  }
  if(g_diag.current_mode==2u) continue;
  if(!ok) break;
  for(unsigned bus=0;bus<3;bus++) {masks[bus]=prepare_can(bus,&replies[bus]);g_diag.virtual_replies++;}
  if(!masks[0]||!masks[1]||!masks[2]) {ok=false;break;}
  if(recording) {capture(leg*2u+1u,masks);captured_rounds++;} else trigger(masks);
  for(unsigned bus=0;bus<3;bus++) {
   if(!finish_can(bus,&replies[bus],masks[bus],&received[bus])) ok=false;
   else if(!qd_ak_decode_reply(g_diag.current_profile,&received[bus],&virtual_state)) {g_diag.virtual_errors++;ok=false;}
   else g_diag.virtual_decoded++;
  }
  if(!ok) break;
 }
 g_diag.commands_routed++;
 uint32_t elapsed=DWT->CYCCNT-begin;if(elapsed>g_diag.max_route_cycles) g_diag.max_route_cycles=elapsed;
 if(recording) {
  g_diag.capture_sequence=sequence;g_diag.capture_samples_per_round=CAPTURE_SAMPLES;
  g_diag.capture_rounds=captured_rounds;g_diag.capture_requested=0;__DMB();
  g_diag.capture_done=ok&&captured_rounds==CAPTURE_ROUNDS;
 }
 if(!ok && g_diag.current_mode!=2u) qd_controller_fault(&controller,QD_FAULT_IO);
}
static void send_frame(const qd_frame *f) {
 uint8_t bytes[QD_MAX_FRAME]; size_t n=qd_encode_frame(f,bytes,sizeof(bytes));
 if(!n||HAL_UART_Transmit(&uart,bytes,(uint16_t)n,20)!=HAL_OK) {g_diag.tx_errors++;qd_controller_fault(&controller,QD_FAULT_IO);}
}
static void handle_frame(void *u,const qd_frame *f) {
 uint32_t callback_begin=DWT->CYCCNT;
 (void)u; qd_frame reply; bool changed=f->type==QD_HELLO&&f->session!=controller.session;
 enum qd_result status=qd_controller_on_frame(&controller,f,HAL_GetTick(),&reply);
 if(changed&&status==QD_REPLY) {
  g_diag.session=f->session;g_diag.frames_received=0;g_diag.commands_received=0;g_diag.commands_accepted=0;
  g_diag.commands_routed=0;g_diag.route_rejected=0;g_diag.feedback_sent=0;g_diag.missed_ticks=0;g_diag.can_errors=0;
  g_diag.virtual_replies=0;g_diag.virtual_decoded=0;g_diag.virtual_errors=0;g_diag.max_route_cycles=0;
  memset(&virtual_state,0,sizeof(virtual_state));
  memset((void *)g_diag.bus,0,sizeof(g_diag.bus));memset((void *)g_diag.joint,0,sizeof(g_diag.joint));
  for(unsigned b=0;b<3;b++) {g_diag.bus[b].cccr=can[b].Instance->CCCR;g_diag.bus[b].nbtp=can[b].Instance->NBTP;g_diag.bus[b].test_reg=can[b].Instance->TEST;}
 }
 g_diag.frames_received++;
 if(f->type==QD_COMMAND&&f->session==controller.session) {
  g_diag.commands_received++;
  if(status==QD_ACCEPTED) {
   g_diag.commands_accepted++;
   if(!controller.command.enabled) route(&controller.command,f->sequence);
  }
 }
 if(status==QD_REPLY) {reply.payload[3]=0x80;send_frame(&reply);}
 callback_cycles+=DWT->CYCCNT-callback_begin;
}
int main(void) {
 memset((void *)&g_diag,0,sizeof(g_diag));g_diag.magic=0x4743414eu;g_diag.version=3;g_diag.boot_stage=1;
 g_diag.requested_mode=1;g_diag.current_mode=0xffffffffu;
 HAL_Init();clock_init();g_diag.boot_stage=2;
 CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT=0;DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
 qd_board board={NULL,never_ready,synthetic_read,never_apply,disarm};
 qd_controller_init(&controller,&board);qd_parser_init(&parser);
 uart_init();can_init(g_diag.requested_mode);g_diag.boot_stage=3;
 uint32_t next=HAL_GetTick()+5u;uint64_t logical=0;
 while(1) {
  uint32_t loop_begin=DWT->CYCCNT;
  g_diag.current_profile=g_diag.requested_profile==1u?1u:0u;
  if(g_diag.requested_mode!=g_diag.current_mode) {
   if(g_diag.requested_mode>2u) g_diag.requested_mode=1;
   can_init(g_diag.requested_mode);
  }
  qd_controller_tick(&controller,HAL_GetTick());
  if(g_diag.rx_overflows||g_diag.uart_errors||g_diag.tx_errors) qd_controller_fault(&controller,QD_FAULT_IO);
  uint8_t bytes[64];size_t n=0;
  while(rx_tail!=rx_head&&n<sizeof(bytes)) {__DMB();bytes[n++]=rx_ring[rx_tail];__DMB();rx_tail=(rx_tail+1u)&(RX_SIZE-1u);}
  uint32_t queued=(rx_head-rx_tail)&(RX_SIZE-1u);if(queued>g_diag.max_rx_queued) g_diag.max_rx_queued=queued;
  if(n) {
   uint32_t parse_begin=DWT->CYCCNT;callback_cycles=0;
   qd_parser_feed(&parser,bytes,n,handle_frame,NULL);
   uint32_t parse_cycles=DWT->CYCCNT-parse_begin-callback_cycles;
   if(parse_cycles>g_diag.max_parser_cycles) g_diag.max_parser_cycles=parse_cycles;
  }
  g_diag.parser_rejected=parser.rejected_frames;
  uint32_t now=HAL_GetTick();
  if((int32_t)(now-next)>=0) {
   uint32_t ticks=1u+(now-next)/5u;g_diag.missed_ticks+=ticks-1u;next+=ticks*5u;logical+=(uint64_t)ticks*5000000u;
   qd_frame f;if(qd_controller_feedback(&controller,logical,&f)) {send_frame(&f);g_diag.feedback_sent++;}
  }
  uint32_t loop_cycles=DWT->CYCCNT-loop_begin;if(loop_cycles>g_diag.max_loop_cycles) g_diag.max_loop_cycles=loop_cycles;
 }
}

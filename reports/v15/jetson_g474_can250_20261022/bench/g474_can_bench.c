/* Bench-only CAN timing fixture for NUCLEO-G474RE (2026-10-22). Not motor firmware.
 * Clock and FDCAN pins are those of firmware/stm32g474_usb/board/g474_ak_firmware.c.
 * 1. Interlock: in normal mode each bus sends ONE frame with a non-motor ID and must
 *    get an ACK error (no node listening at this rate). Otherwise the bench stops.
 * 2. External loopback: each bus drives its real TX pin and transceiver and receives
 *    its own frames; ACK errors are ignored. Bus 1 sends standard 8-byte frames
 *    (LEGACY MIT shape), buses 2 and 3 extended 8-byte frames (V3 MIT shape).
 *    burst: 1000 back-to-back frames per bus, all buses at once.
 *    schedule: every 5 ms, 8 frames per bus (4 commands + 4 replies of one bus) for 10 s.
 *    Pattern 0 has all-zero data, pattern 1 xorshift data.
 * Results are read over SWD from g_bench. */
#include "stm32g4xx_hal.h"
#include <stdbool.h>
#include <string.h>

#ifndef G474_CAN_KBPS
#define G474_CAN_KBPS 250u
#endif
#define CAN_PRESCALER (5000u/G474_CAN_KBPS)
_Static_assert(CAN_PRESCALER*G474_CAN_KBPS==5000u&&CAN_PRESCALER<=512u,"unsupported CAN bit rate");
#define CPU_HZ 160000000u
#define BURST 1000u
#define CYCLES 2000u
#define PERIOD (CPU_HZ/200u)
#define PER_CYCLE 8u
#define BENCH_MAGIC 0x43414e42u /* "CANB" */

typedef struct {
 uint32_t magic, version, kbps, stage, faults, tx_errors, rx_errors;
 uint32_t nbtp[3], interlock_psr[3], interlock_rx[3], interlock_ok;
 /* [pattern][bus] */
 uint32_t burst_frames[2][3], burst_cycles[2][3], burst_ts_bits[2][3];
 uint32_t sched_cycles[2], sched_late_start[2], sched_max_cycles_late[2];
 uint32_t sched_samples[2][3], sched_max_cycles[2][3], sched_sum_us[2][3], sched_overrun[2][3];
} g474_can_bench;
_Static_assert(sizeof(g474_can_bench)==65u*4u,"read_bench.py layout");
volatile g474_can_bench g_bench;
static FDCAN_HandleTypeDef can[3];
static uint32_t rng=0x2545f491u;

static void fatal(void) { g_bench.stage |= 0x80000000u; __disable_irq(); while(1) __WFI(); }
void HardFault_Handler(void) { fatal(); }
void MemManage_Handler(void) { fatal(); }
void BusFault_Handler(void) { fatal(); }
void UsageFault_Handler(void) { fatal(); }
void SysTick_Handler(void) { HAL_IncTick(); }
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
 RCC_PeriphCLKInitTypeDef p={0}; p.PeriphClockSelection=RCC_PERIPHCLK_FDCAN;
 p.FdcanClockSelection=RCC_FDCANCLKSOURCE_PLL;
 if(HAL_RCCEx_PeriphCLKConfig(&p)!=HAL_OK) fatal();
 if(HAL_RCC_GetHCLKFreq()!=CPU_HZ) fatal();
}
static void pins_init(void) {
 __HAL_RCC_GPIOA_CLK_ENABLE(); __HAL_RCC_GPIOB_CLK_ENABLE(); __HAL_RCC_FDCAN_CLK_ENABLE();
 CLEAR_BIT(DBGMCU->CR,DBGMCU_CR_TRACE_IOEN);
 GPIO_InitTypeDef p={0}; p.Mode=GPIO_MODE_AF_PP; p.Pull=GPIO_PULLUP; p.Speed=GPIO_SPEED_FREQ_VERY_HIGH;
 p.Pin=GPIO_PIN_11|GPIO_PIN_12; p.Alternate=GPIO_AF9_FDCAN1; HAL_GPIO_Init(GPIOA,&p);
 p.Pin=GPIO_PIN_12|GPIO_PIN_13; p.Alternate=GPIO_AF9_FDCAN2; HAL_GPIO_Init(GPIOB,&p);
 p.Pin=GPIO_PIN_8|GPIO_PIN_15; p.Alternate=GPIO_AF11_FDCAN3; HAL_GPIO_Init(GPIOA,&p);
}
static void can_start(uint32_t mode) {
 for(unsigned i=0;i<3;i++) {
  can[i].Instance=i==0?FDCAN1:(i==1?FDCAN2:FDCAN3);
  FDCAN_InitTypeDef *c=&can[i].Init;
  c->ClockDivider=FDCAN_CLOCK_DIV2; c->FrameFormat=FDCAN_FRAME_CLASSIC; c->Mode=mode;
  c->AutoRetransmission=DISABLE; c->TransmitPause=DISABLE; c->ProtocolException=DISABLE;
  c->NominalPrescaler=CAN_PRESCALER; c->NominalSyncJumpWidth=2; c->NominalTimeSeg1=13; c->NominalTimeSeg2=2;
  c->DataPrescaler=5; c->DataSyncJumpWidth=2; c->DataTimeSeg1=13; c->DataTimeSeg2=2;
  c->StdFiltersNbr=0; c->ExtFiltersNbr=0; c->TxFifoQueueMode=FDCAN_TX_FIFO_OPERATION;
  if(HAL_FDCAN_Init(&can[i])!=HAL_OK) fatal();
  if(HAL_FDCAN_ConfigGlobalFilter(&can[i],FDCAN_ACCEPT_IN_RX_FIFO0,FDCAN_ACCEPT_IN_RX_FIFO1,FDCAN_REJECT_REMOTE,FDCAN_REJECT_REMOTE)!=HAL_OK) fatal();
  /* Rx timestamps count nominal bit times. */
  if(HAL_FDCAN_ConfigTimestampCounter(&can[i],FDCAN_TIMESTAMP_PRESC_1)!=HAL_OK) fatal();
  if(HAL_FDCAN_EnableTimestampCounter(&can[i],FDCAN_TIMESTAMP_INTERNAL)!=HAL_OK) fatal();
  if(HAL_FDCAN_Start(&can[i])!=HAL_OK) fatal();
  g_bench.nbtp[i]=can[i].Instance->NBTP;
 }
}
static void can_stop(void) {
 for(unsigned i=0;i<3;i++) if(HAL_FDCAN_Stop(&can[i])!=HAL_OK||HAL_FDCAN_DeInit(&can[i])!=HAL_OK) fatal();
}
static uint32_t xorshift(void) { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
/* Non-motor IDs: standard 0x700+n, extended 0x1f0008f0+n. */
static bool queue(unsigned b,unsigned n,unsigned pattern) {
 if(HAL_FDCAN_GetTxFifoFreeLevel(&can[b])==0u) return false;
 FDCAN_TxHeaderTypeDef t={0}; uint8_t d[8]={0};
 t.IdType=b?FDCAN_EXTENDED_ID:FDCAN_STANDARD_ID; t.Identifier=b?0x1f0008f0u+(n&7u):0x700u+(n&7u);
 t.TxFrameType=FDCAN_DATA_FRAME; t.DataLength=FDCAN_DLC_BYTES_8; t.ErrorStateIndicator=FDCAN_ESI_ACTIVE;
 t.BitRateSwitch=FDCAN_BRS_OFF; t.FDFormat=FDCAN_CLASSIC_CAN; t.TxEventFifoControl=FDCAN_NO_TX_EVENTS;
 if(pattern) { uint32_t a=xorshift(),c=xorshift(); memcpy(d,&a,4); memcpy(d+4,&c,4); }
 if(HAL_FDCAN_AddMessageToTxFifoQ(&can[b],&t,d)!=HAL_OK) { g_bench.tx_errors++; return false; }
 return true;
}
static bool take(unsigned b,uint16_t *ts) {
 uint32_t fifo=b?FDCAN_RX_FIFO1:FDCAN_RX_FIFO0;
 if(HAL_FDCAN_GetRxFifoFillLevel(&can[b],fifo)==0u) return false;
 FDCAN_RxHeaderTypeDef r; uint8_t d[8];
 if(HAL_FDCAN_GetRxMessage(&can[b],fifo,&r,d)!=HAL_OK) { g_bench.rx_errors++; return false; }
 *ts=(uint16_t)r.RxTimestamp; return true;
}
static void wait_cycles(uint32_t n) { uint32_t s=DWT->CYCCNT; while(DWT->CYCCNT-s<n) {} }

static bool interlock(void) {
 can_start(FDCAN_MODE_NORMAL);
 for(unsigned b=0;b<3;b++) if(!queue(b,0,0)) fatal();
 wait_cycles(CPU_HZ/100u); /* 10 ms, far beyond one frame at 125 kbit/s */
 bool ok=true;
 for(unsigned b=0;b<3;b++) {
  uint32_t psr=can[b].Instance->PSR; g_bench.interlock_psr[b]=psr;
  g_bench.interlock_rx[b]=HAL_FDCAN_GetRxFifoFillLevel(&can[b],FDCAN_RX_FIFO0)+HAL_FDCAN_GetRxFifoFillLevel(&can[b],FDCAN_RX_FIFO1);
  ok=ok&&(psr&FDCAN_PSR_LEC)==FDCAN_PROTOCOL_ERROR_ACK&&g_bench.interlock_rx[b]==0u;
 }
 can_stop();
 return ok;
}
static void burst(unsigned p) {
 uint32_t sent[3]={0},got[3]={0},first[3]={0},last[3]={0},bits[3]={0}; uint16_t prev[3]={0};
 uint32_t begin=DWT->CYCCNT;
 while(got[0]<BURST||got[1]<BURST||got[2]<BURST) {
  for(unsigned b=0;b<3;b++) {
   while(sent[b]<BURST&&queue(b,sent[b],p)) sent[b]++;
   uint16_t ts;
   while(take(b,&ts)) {
    uint32_t now=DWT->CYCCNT;
    if(got[b]) bits[b]+=(uint16_t)(ts-prev[b]); else first[b]=now;
    prev[b]=ts; last[b]=now; got[b]++;
   }
  }
  if(DWT->CYCCNT-begin>5u*CPU_HZ) { g_bench.faults++; break; }
 }
 for(unsigned b=0;b<3;b++) {
  g_bench.burst_frames[p][b]=got[b]; g_bench.burst_cycles[p][b]=last[b]-first[b]; g_bench.burst_ts_bits[p][b]=bits[b];
 }
}
/* A cycle completes when its 8th frame is received. A cycle that starts while the
 * previous one is still on the bus is an overrun and gives no completion sample. */
static void schedule(unsigned p) {
 uint32_t start[3]={0},tx[3]={0},rx[3]={0},n[3]={0}; bool valid[3]={0};
 uint32_t next=DWT->CYCCNT+PERIOD,k=0,tail=0;
 while(k<CYCLES||rx[0]||rx[1]||rx[2]) {
  uint32_t now=DWT->CYCCNT;
  if(k<CYCLES&&(int32_t)(now-next)>=0) {
   uint32_t late=now-next;
   if(late>g_bench.sched_max_cycles_late[p]) g_bench.sched_max_cycles_late[p]=late;
   if(late>CPU_HZ/10000u) g_bench.sched_late_start[p]++; /* > 100 us */
   for(unsigned b=0;b<3;b++) {
    if(rx[b]) g_bench.sched_overrun[p][b]++;
    valid[b]=rx[b]==0u; start[b]=next; tx[b]+=PER_CYCLE; rx[b]+=PER_CYCLE;
   }
   next+=PERIOD; k++; g_bench.sched_cycles[p]=k;
   if(k==CYCLES) tail=now;
  }
  for(unsigned b=0;b<3;b++) {
   while(tx[b]&&queue(b,n[b],p)) { tx[b]--; n[b]++; }
   uint16_t ts;
   while(take(b,&ts)) {
    if(!rx[b]) { g_bench.rx_errors++; continue; }
    if(--rx[b]==0u&&valid[b]) {
     uint32_t c=DWT->CYCCNT-start[b];
     g_bench.sched_samples[p][b]++; g_bench.sched_sum_us[p][b]+=c/(CPU_HZ/1000000u);
     if(c>g_bench.sched_max_cycles[p][b]) g_bench.sched_max_cycles[p][b]=c;
    }
   }
  }
  if(k==CYCLES&&DWT->CYCCNT-tail>CPU_HZ/10u) { g_bench.faults++; break; }
 }
}
int main(void) {
 memset((void *)&g_bench,0,sizeof(g_bench));
 g_bench.magic=BENCH_MAGIC; g_bench.version=1; g_bench.kbps=G474_CAN_KBPS; g_bench.stage=1;
 HAL_Init(); clock_init();
 CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT=0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
 pins_init(); g_bench.stage=2;
 g_bench.interlock_ok=interlock();
 if(!g_bench.interlock_ok) { g_bench.stage=0x10u; while(1) __WFI(); }
 can_start(FDCAN_MODE_EXTERNAL_LOOPBACK); g_bench.stage=3;
 burst(0); burst(1); g_bench.stage=4;
 schedule(0); schedule(1); g_bench.stage=5;
 can_stop(); /* bus idle after the bench */
 g_bench.stage=6;
 while(1) __WFI();
}

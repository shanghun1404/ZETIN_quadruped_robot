# V16 보행 디버깅 기록

2026-09-29. 실행 완료와 보행 성공을 구분한다. 모든 시뮬레이션은 HPC Slurm에서 수행했다.
원본 V15 정책은 수정하지 않았다. 별도 물리 제어기의 결함과 실패 자료를 보존한다.

| 작업 | 조건 | 실제 결과 |
|---|---|---|
| 527 | 모터 관성 시험·5초 기립, 4환경 | 기립4/4; armature 포함 무접촉 가속도 상대 오차 최대8.67e-7 |
| 528 | +X, 64환경×3세대, 6초 CEM | 모든 세대 생존0/64 |
| 529/530 | +Y 평가·pilot | GPU1 VM 관리 DRAIN 동안 실행 전 취소; 결과 없음 |
| 531 | frozen 이전 코드, +Y/.25m/s, 8환경×8초 | 생존0/8 |
| 533 | duty.82/period1.6s, foot COM 가속도 투영, .1m/s, 8환경×12초 | 생존0/8 |
| 535 | 동일 주기+COM 준비/착지 확인 FSM, 8환경×12초 | 생존0/8; 모두첫lift1회, 완료step0회 |
| 537 | 느린주기4s/낮은발높이.02m/깊은COM inset.055m, .05m/s, 8환경×16초 | 생존0/8; 첫lift1회, 완료step0회 |

## 확인한 결함과 실험 근거

1. native PhysX 관절 순서는 canonical 순서와 다르다. 관절 이름으로 q/dq/M/h/J/torque를 모두 변환한다.
2. source fore-aft는 +Y다. 527 nominal Jacobian에서 HAA는 주로X, HFE/KFE는 주로Y 운동을 만든다.
   초기 V16 standalone의 +X 설정을 수정했으며 V15 policy의 진행 축을 변경한 것은 아니다.
3. 초기 quarter-offset duty.70에서는 두 swing이 겹친다. 새범위 duty.80–.90은 scheduled stance≥3을 보장한다.
4. 531 env0 첫FL liftoff(t1.60)에서 전체COM의 다음세발 삼각형 margin은−.0392m, RR normal force는0이었다.
   static polygon 진단만으로 동적 fall의 원인을 확정하지는 않으나 COM 준비가 없는 구조를 개선할 근거다.
5. 533 요청 가속도에 대한 stance 제약 잔차는 약1e-4m/s² 수준이지만 실제 foot slip/fall이 남았다.
   수학적 투영 잔차가 작다고 실제 solver 접촉이 만족된 것은 아니다.
6. 535는 실제COM을 옮겼다. env0 첫lift(t1.48) margin.028m, RR normal66.4N이었으나
   t1.72에는 RR normal0으로 감소했다. 모든 환경에서 다음step으로 넘어가기 전에 실패했다.
   빠른swing의 관성 효과, 요청 base 가속도의 투영 손실, 접촉점 근사가 다음 조사 대상이다.
   어느 하나를 검증된 유일 원인으로 표현하지 않는다.

현재 prepared crawl은 `PREPARE → SWING → SETTLE`이고 readiness/contact dwell 동안 phase를 멈춘다.
COM 목표는 whole-robot COM 기준이며 다음 지지삼각형 inset 안으로 요청점을 최소 이동 투영한다.
요청 목표는 기다리는 동안 계속 멀어지지 않도록 한 step 동안 고정된다.
실제 접촉은 foot net normal force>20N proxy다. 527 기립의 발별 normal force는 약93–130N이었다.
정확한 collision contact patch를 측정한 판정은 아니다.

## 실행 및 회귀

532: 455 passed /3 skipped. 534: closest COM target/FSM와 Torch parity 포함 **463 passed /3 skipped**.
이는 소스/수학 검증이며 보행 성공 검증과 구분한다.
각 GPU 작업의 `job_sources/<id>/`는 실제 실행 소스 snapshot이고 manifest가 hash를 기록한다.
local 원자료: [physics_jobs](physics_jobs/). HPC: `/home/shanghun1404/v16_12dof_model/physics_jobs/`.

01:36 root의 GPU1 VFIO teardown으로 node DRAIN, 01:57:58 관리 작업 완료 후 GPU2/58GB로 복귀했다.
관련 관리 설정은 수정하지 않았다. 536 V15 적응 훈련과537 V16 평가가 각기 다른 GPU UUID에서 실행됨을 확인했다.
536은 guard.97/LPF23Hz, fixedLR3.375e-5, optimizer state 보존,512환경×250회다.
새 정책의 채택은 기존339와 동일한 행동·토크 gate 및 독립 seed 평가에 달려 있다.

## QP와 착지 후 정지 (550/551)

549는 installed Isaac Sim5.1의 static/dynamic/viscous 세 값이 모두0임을 기록했고, LS 배분에서는0/8 보행 생존이었다.
550/551은 같은 조건에서 힘 배분만 QP로 바꿨다. 550 기립4/4,551 12초생존8/8이다.
551의 힘/모멘트 평균 잔차는 몸무게 및 몸무게×.30m 기준 약1.1e-5/1.4e-5이고,
토크 bound violation과 guard 활동은0이다. 600 control tick 중1개가1e-3 numerical tolerance를조금넘었다.
그러나 모두liftoff1회/완료step0회, 진행약.012m, 속도약.0012m/s이므로보행 gate를통과하지못했다.

첫발FL은t약3초부터지면에닿아2–4N normal force를보였다.
active force/constraint mask도FSM의20N 하중조건을사용해FL을계속제외했고,FL에하중을배분할기회가없었다.
다음수정은접촉존재>1N을force/constraint에,충분한하중>20N을FSM readiness에사용한다.
새rigid local contact point로relatch한tick은이전point와J를차분하지않는다.

548 가속도 진단은 post-step ground normal+friction F와 기존 latched point J를 사용하는 근사다.
HFE/KFE의 paired 잔차는 contact couple/작용점 오차를 포함할 수 있어 joint friction으로 단정하지 않는다.
measured 모드에서 매tick 압력중심을 바꾸는 것은 rigid material point의 Jdot와 다르다.
이모드는기록된진단실험이며현재준비보행은 ground local point를유지한다.


## 552 이후: 접촉·목표전환·반복 훈련

552/553은presence>1N으로착지하중의순환조건을수정했다.16초생존은2/8·4/8이었다.
다음세발삼각형의margin/target오차로정지했고, 기존projection은몸통요청을거의0으로바꿨다.
첫재접촉가속도요청일부는보수적인마찰피라미드의aggregate bound를넘었다.
이는현재피라미드에서의고정가속도불일치이며,mu.6원형cone전체의불가능성증명은아니다.

557에서baseweight1000을시험한뒤,COM목표전환을minimumjerk로연결했다.
561은완료된reference의endpoint에서만새segment를시작하고8/8생존·5–9steps를확인했다.
563은128환경×24세대,568은128후보×4반복×12세대를완료했다.
568최종선택후보는4회최소진행.751m·완료9steps,최종population512/512생존이다.
571추가200세대가실행중이고,582/573독립평가와581영상은의존성으로연결했다.

2026-09-29 04:10 KST 운영 리뷰에서 Slurm이 제출 당시 batch script를 보존하는 것을 확인했다.
이전572는 평가 길이 metadata가 없고574는 training checkpoint를 영상 폴더에 복사하지 않았다.
현재 summarizer/observer와 호환되지 않아 두 작업을 실행 전에 취소하고 최신582/581로 교체했다.
573의32초 평가와581 영상의 실제 의존성도 afterok582로 수정했다. 훈련571/577과 제어기 소스는 그대로 진행한다.

566WBC기립4/4였으나570보행은0/8,동일QP는8/8이었다.
초기WBC발궤적추종지연/overshoot가수치infeasible보다앞서나타났다.
575는가속도task가중치를변수스케일에서보정하고full18 a와primal/dual잔차를추가한비교다.
기존주훈련의소스/분포를변경하지않았다.

569는첫QP실행40초에출력override와자동analyzer경로의불일치를발견해취소했다.
V16_OUTPUT_ROOT로runtime/analyzer경로를통일한570은완료0:0이다.
현재실행·완료·제약위반·평가대기는[overnight_status.md](overnight_status.md)에기록한다.

## 2026-09-29 영상 출력 폴더 수정

581/580은 렌더링 앱 시작 뒤 Plant의 `mkdir(exist_ok=False)`에서 실패했다.
영상 wrapper가 같은 출력 폴더에 checkpoint를 미리 넣은 것이 직접 원인이다.
준비 파일을 `video_inputs/<job>/`로 분리하고 rollout이 새 출력 폴더를 만든 뒤
checkpoint·실행 인자를 복사하도록 wrapper를 수정했다. 원본 훈련 제어기 소스는 그대로 사용한다.
QP571과 WBC577의 최종 모델로583/584를 제출했다.

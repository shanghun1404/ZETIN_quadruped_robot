# V15 Jetson / STM32G474 복사 준비 묶음

대상은 Jetson Orin Nano Super의 Ubuntu 24.04/aarch64, ROS2 Jazzy와 NUCLEO-G474RE/MB1367이다. 이번 산출물은 나중에 복사할 소프트웨어와 MCU 포팅 소스다. Jetson 설치, 보드 플래싱, 실제 USB 지연 및 모터 구동은 별도의 실기 검증이 필요하다.

`model_15699.pt`의 SHA256은 `9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6`, 동결 V12 student의 SHA256은 `74c0c15644aec802c2637e5054286ea261e0fd6060637506eee61776af3b6f05`다. `manifest.json`은 실제 복사한 파일의 해시, 원본 저장소 HEAD, CAD 물리 모델과 기존 시뮬레이션 보고서의 식별 정보를 담는다. 작업 디렉터리의 미커밋 소스도 파일별 SHA로 식별한다.

기존 ROS2/Isaac Sim 후보 검증과 새 USB 연결 검증은 범위가 다르다. 포함한 `evidence/reports/v15/cad_rms_20261011/round2/summary.json`의 결과를 실제 Jetson/MCU/모터의 결과로 해석하지 않는다. CAD의 관절 부호를 실제 배선 부호로 사용할 수 없다.

## 저장소에서 번들 생성

저장소 루트에서 동결 student 파일을 명시한다. builder는 student·actor 해시가 다르거나 필요한 소스가 없으면 기존 압축 파일을 보존한 채 실패한다.

```bash
python3 deploy/jetson_v15/build_bundle.py \
  --student-checkpoint /path/to/trusted/v12_distillation_dagger/best.pt
```

기본 출력은 `reports/v15/jetson_usb_20261012/v15_jetson_20261012.tar.gz`다. 빌더에는 ROS2, Torch, Isaac Sim 설치가 필요 없다. 같은 파일 내용과 저장소 HEAD에서는 파일 시간과 출력 파일명에 관계없이 같은 압축 바이트를 생성한다. 번들을 바꾸면 다시 빌드한다.

기존 `v12_camera.make_v12_depth_camera_cfg` 및 `v13_depth_policy.make_v13_depth_wrapper`의 함수 내부 훈련용 IsaacLab import는 소스에 보존하고 manifest에 별도로 기록한다. 하드웨어 정책은 이 두 factory를 호출하지 않는다. 모듈 최상위의 simulator import와 미확정 로컬 의존 모듈은 번들 생성 오류로 처리한다.

포함 구조:

```text
v15_jetson_20261012/
  workspace/src/{v14_ros2_control,v151_ros2_control,v15_hardware_bridge}/
  runtime/config/          # 실행 모듈과 재귀적으로 발견한 로컬 의존 모듈
  model/                   # 원본 CAD dynamics와 metadata
  checkpoints/             # actor와 동결 depth student
  firmware/stm32g474_usb/  # C 규약 및 G474 연결 예제
  evidence/                # 설계, 계획, 기존 후보 검증 식별 자료
  manifest.json, SHA256SUMS
  check_env.py, verify_bundle.py, build.sh, run.sh
```

## Jetson에서 환경 기록 및 의존 패키지 준비

압축을 푼 뒤 다음 명령은 파일 해시를 확인하고 실제 OS/CPU/ROS2와 `/etc/nv_tegra_release`의 BSP 정보를 기록한다. 환경 검사는 OS를 변경하지 않는다. BSP 파일이 없으면 JSON에 `null`로 남으며 CUDA wheel 선택은 미확정이다.

```bash
tar -xzf v15_jetson_20261012.tar.gz
cd v15_jetson_20261012
python3 verify_bundle.py
source /opt/ros/jazzy/setup.bash
python3 check_env.py --output environment.json
```

ROS2 Jazzy의 Ubuntu 24.04/arm64 deb와 저장소 설정은 [공식 설치 문서](https://repo.test.ros2.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)를 따른다. ROS 저장소를 설정한 환경에서 아래는 사용자가 실행할 패키지 설치 예시다. 번들 스크립트는 apt나 sudo를 실행하지 않는다.

```bash
sudo apt install ros-jazzy-ros-base ros-jazzy-pinocchio \
  ros-jazzy-launch-ros ros-jazzy-sensor-msgs ros-jazzy-nav-msgs \
  ros-jazzy-geometry-msgs ros-jazzy-std-msgs \
  python3-colcon-common-extensions python3-venv python3-pip \
  python3-numpy python3-serial python3-yaml
```

가상환경은 `/usr/bin/python3`로 만들고 시스템 패키지를 보이게 한다. deb ROS2와 같은 Python ABI를 유지해야 한다는 [ROS2 공식 Python 문서](https://repo.test.ros2.org/en/jazzy/How-To-Guides/Using-Python-Packages.html)에 따른다. Conda의 별도 Python을 사용하면 ROS2 바이너리 ABI가 맞지 않을 수 있다.

```bash
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -c 'import rclpy, numpy, torch, pinocchio, serial; print(torch.__version__, pinocchio.__version__)'
python3 -m pip freeze > installed-python-packages.txt
```

CPU wheel은 [PyTorch 공식 CPU 설치 경로](https://pytorch.org/get-started/locally/)로 시작한다. 해당 Python/aarch64 조합에 wheel이 없다면 설치 오류를 해결한 뒤 진행한다. GPU wheel은 기록한 BSP와 CUDA/Python 조합에 맞춰 나중에 선택한다. CPU에서 정책 50 Hz와 관절 송신 200 Hz의 실측 지연을 먼저 확인해야 한다.

Pinocchio는 [공식 저장소의 ROS 설치 안내](https://github.com/stack-of-tasks/pinocchio#ros)에 따라 `ros-jazzy-pinocchio`의 Python binding을 사용한다. ROS 패키지를 쓸 수 없는 경우 같은 시스템 Python 가상환경에서 `python3 -m pip install --only-binary=:all: pin`을 사용할 수 있으나 aarch64 wheel과 NumPy ABI를 먼저 확인한다. PyPI 배포명은 `pin`, import 이름은 `pinocchio`다.

## 빌드와 기본 비활성 실행

```bash
bash build.sh
bash run.sh
```

`build.sh`는 환경 검사, 파일 해시와 실행 의존 import를 확인한 뒤 실제 colcon workspace에서 세 ROS2 패키지를 빌드한다. 가상환경 Python으로 colcon을 실행하여 생성된 노드가 같은 Python을 사용하게 한다. `run.sh`는 번들 위치에서 actor/student/model/config의 절대 경로를 만들며 원격 ws1 디렉터리나 Isaac Sim에 의존하지 않는다. 빌드 후 번들 위치를 옮겼다면 생성된 workspace와 가상환경을 새 위치에서 다시 만든다.

기본 `enable_motors=false`다. 새 패키지는 초기 SIT 및 정지 명령으로 시작한다. 실제 모터 CAN/MIT 인코딩, 12개 모터 ID, 부호, 영점, 전원/enable 회로, 측정값 단위 변환은 보드의 motor callback에 연결해야 한다. `firmware/stm32g474_usb/`의 보드 포팅 안내를 따라 호스트의 검증 표시와 MCU의 보정 게이트를 모두 완성해야 한다.

실기 보정값은 원본 번들 설정을 보존하고 별도 JSON에 넣을 수 있다. 실제 검증을 끝낸 이후 명시적으로 활성화할 때만 다음 형태를 사용한다.

```bash
cp workspace/src/v15_hardware_bridge/config/hardware.json /path/to/robot-hardware.json
V15_HARDWARE_CONFIG=/path/to/robot-hardware.json V15_ENABLE_MOTORS=true bash run.sh
```

`V15_HARDWARE_CONFIG`는 gateway 설정 경로이며 STM32의 CAN 어댑터 및 부호/영점 구현을 대신하지 않는다. 보정은 MCU 모터 경계에서 한 번 적용한다. 호스트가 이미 CAD/policy canonical SI 데이터를 보내므로 호스트에서 중복 보정하지 않는다.

USB의 Kp≤300, Kd≤8은 제어기의 명령 범위다. 실제 AK70-10/AK10-9 KV60 V3.0의 firmware와 CAN/MIT 게인 인코딩 범위는 별도로 확인한다. 과거 모터 매뉴얼의 Kd 상한을 V3.0에 그대로 적용하거나, 범위를 맞추려고 Kd를 조용히 잘라서는 안 된다. 지원하지 않는 게인은 거부하고 실제 모터 계약을 확정한다.

## 실제 센서와 시각 계약

현재 actor는 84×84 깊이 영상과 동결 student를 사용한다. 깊이 카메라 없이 이 후보의 관측을 구성할 수 없다. 실제 센서 드라이버는 bundle에 포함되지 않는다. 아래 raw topic에 실제 측정 시각을 가진 입력을 공급해야 한다.

| topic | 입력 |
|---|---|
| `raw/joint_states` | MCU gateway의 12관절 위치/속도/토크 |
| `raw/odom` | 실제 추정기 `nav_msgs/Odometry`, `child_frame_id=base`, body 축 twist |
| `raw/depth/image` | 84×84 깊이(m), little-endian `32FC1`, step=336 |
| `raw/foot_contacts` | `sensor_msgs/JointState`, FL/FR/RL/RR `_foot` 이름과 `effort`의 측정 접촉력(N) |

기본 namespace는 `v15real`이므로 외부 센서 드라이버는 `/v15real/raw/odom` 같은 전체 topic 경로를 사용한다. 다른 namespace는 `bash run.sh namespace:=이름`으로 정한다. 오도메트리의 quaternion은 정규화되어야 하고 발 접촉력은 유한한 비음수여야 한다.

5 ms MCU 틱에 센서를 모으고 `use_sim_time=false`를 사용한다. odom/발 접촉은 75 ms, 깊이는 200 ms, MCU 피드백은 50 ms 이내여야 한다. 10 ms를 넘는 미래 시각과 틱 누락은 거부한다. 빠진 입력을 가짜 데이터로 보충하지 않는다. 외부 오도메트리 좌표와 발 순서는 새 패키지의 센서 계약에 맞춘다.

USB는 STLINK-V3E VCP→LPUART1 PA2(TX)/PA3(RX), 921600 baud/8N1을 가정한다. 각 방향 200 Hz의 실제 처리량과 버퍼 지연은 보드에서 검증한다. 연결이 복구되어도 자동 arm되지 않는다. 호스트 50 ms stale 및 MCU 100 ms watchdog과 STOP/disarm 동작을 실제 장치에서 먼저 확인한다. 자동 시작 서비스는 설치하지 않는다.

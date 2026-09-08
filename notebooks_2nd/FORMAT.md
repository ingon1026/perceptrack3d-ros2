# notebooks_2nd — "AI 이전 실무 방식" 재구성 노트북의 공통 형식

목적: 완성본(`notebooks/`, `src/perceptrack3d/`)을 정답지로 두고, 같은 결과를 **실무자가 검색해서 가져다 붙이는 방식**으로 다시 만든다.
각 노트북은 "조각(piece)" 의 나열이다. 조각 = 실무자가 한 번의 검색으로 가져오는 크기(함수 1~2개 또는 문서 예제 1개).

## 노트북 머리
1. 제목 셀: `# Phase N — <제목> (실무 방식 재구성)`
2. "이 노트북에서 가져오는 것" 표: | 조각 | 출처 | 라이선스 | 왜 |
3. 공통 준비 셀: `from perceptrack3d.config import load_config; cfg = load_config()` — 경로는 여기서만. matplotlib `%matplotlib inline`.

## 조각 하나의 셀 구성 (순서 고정)
1. **마크다운** `### 조각 k. <무엇을 가져오나>`
   - `**검색어**`: 실무자가 실제로 칠 검색어 (예: `kitti velodyne bin format python`)
   - `**출처**`: 이름 — URL — 라이선스 — 가져온 날짜(2026-09-08). 파일을 `notebooks_2nd/sources/` 에 저장했으면 그 경로도.
   - `**왜 이걸 가져오나**`: 1~2문장.
   - `**읽을 때 볼 곳**`: 원본에서 어느 줄/어느 절을 보면 되는지 (readme 의 절 이름, 함수 이름).
2. **코드 셀 A — 원본 발췌**: 첫 줄 주석 `# ORIGINAL: <출처 파일> :: <함수/절>, <라이선스>`. 원문 **그대로** (변수명·주석 포함, 영어 주석 유지). MATLAB 원문은 마크다운 코드블록으로 보여 주고, 코드 셀에는 "numpy 로 옮긴 것" 을 `# PORTED FROM ...` 으로 표시.
3. **코드 셀 B — 우리 데이터에 적용**: 원본 함수를 우리 경로·config 로 호출. 원본을 고쳐야 했다면 고친 줄 끝에 `# CHANGED: 이유`. 출력은 shape / dtype / min·max / 첫 몇 값.
4. **코드 셀 C — 검증**: 정답지(`perceptrack3d` 모듈)와 비교. `np.allclose`, shape 동일, 또는 그림 나란히. 다르면 왜 다른지 마크다운으로 설명(예: 좌표 규약 차이).

## 노트북 꼬리
- `### 이어붙이기`: 조각들을 이어 이 Phase 의 최종 함수 하나로 만든다(접착 코드는 직접 씀, 20~50줄). 그 결과를 정답지와 최종 비교.
- `### 여기서 배우는 것`: 3~5 불릿. "남의 코드를 붙일 때 실제로 터지는 곳"(좌표계·shape·dtype·단위) 중심.
- `### 자가 점검 3문제` (답은 접힌 `<details>` 로).

## 규칙
- 마크다운·우리 주석은 한국어, 원본 발췌는 원문 유지.
- 정답지 모듈은 **검증 셀에서만** import 한다. 조각 셀에서는 쓰지 않는다 (직접 붙이는 연습이므로).
- 출처 파일은 `notebooks_2nd/sources/<이름>/` 에 두고 `notebooks_2nd/sources/SOURCES.md` 에 한 줄 추가 (URL, 라이선스, 날짜, 어느 노트북이 쓰는지).
- 라이선스: MIT/Apache/BSD 는 원문 사용. GPL(SORT) 도 사용자 허락으로 원문 발췌하되 라이선스 표기를 반드시 남긴다. 문서 예제(ultralytics/Open3D/scipy)는 문서 URL 표기.
- 실행: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks_2nd/NN_*.ipynb` 로 전체 실행이 오류 없이 끝나야 한다. 그림은 인라인.
- 노트북 생성은 nbformat 으로. 셀 수는 조각 4~7개 × 4셀 + 머리/꼬리 ≈ 25~40 셀.
- 정답지 노트북(`notebooks/`)과 `src/` 는 수정하지 않는다.

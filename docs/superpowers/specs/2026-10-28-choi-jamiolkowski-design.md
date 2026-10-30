# ChoiJamiolkowski 設計案

状態: 採用。ユーザーが「独立コア＋TTDynamics extensionで進める（推奨）」を選択。

## 目的と範囲

`/home/takahashi/.julia/dev/ChoiJamiolkowski` に、同名のJuliaパッケージを作る。
TTDynamicsのTFD HamiltonianとTTSolverのTT伝播を再利用し、系とancillaの
最大エンタングル状態を一度伝播することで、指定時刻のreduced dynamical mapを得る。
構築後は、同じ環境初期状態に対する任意の系初期密度行列へmapを適用できる。

後続のProcess tensor実装で再利用できるよう、Choi表現と伝播バックエンドを分離する。
今回の成果物はパッケージ、APIの説明、実行可能なspin-boson例、独立した数値参照によるテスト。
多時刻介入を含むProcess tensor、HEOMバックエンド、記憶核・transfer tensor・長時間外挿、
FMOの大規模再現計算は初期実装の範囲外。

## 対象論文

参照候補はR. Borrelli and H. Takahashi,
*Reduced Dynamical Maps in Finite Temperature Vibronic Coupling Models via Choi Matrices:
Numerical Methods and Applications*, DOI: https://doi.org/10.1063/5.0332266。
公開著者原稿: https://arxiv.org/html/2605.22459v1。
公開著者原稿は2026年5月21日付で、依頼文の「JCP2025」と年が異なる。
式(6)–(20)の構築法とAppendix AのTT縮約を実装の根拠とする提案。
別の2025年論文を指す場合は実装前に対象を修正する。

## 構成の選択肢

1. **提案: 独立コアとJulia package extension。** コアはLinearAlgebraに依存し、
   TTDynamicsとTTSolverの両方を読み込むとTTバックエンドを有効にする。
   mapの解析だけを行う利用者が伝播ライブラリを読み込まずに済み、
   将来の別バックエンドやProcess tensorとの共通化が容易。
   weakdeps・extensionsとそのロード確認が必要。
2. **TTDynamicsを必須依存とする別パッケージ。** 初期構成は単純だが、
   コアのmap操作も常にTTDynamicsとHEOMKit等に依存する。

両案ともディレクトリ名とパッケージ名は依頼どおり`ChoiJamiolkowski`。
ユーザーは案1を選択。以下はこの構成で実装する設計。

## 数式・表現の規約

- Hilbert空間の脚はoutputとinput（ancilla）を明示し、次元を型の値として保持する。
  初期版でも両次元を区別し、単一時刻mapの表現を多時刻テンソルと混同しない。
- 公開Choi行列はoutput ⊗ inputのKronecker順で、行番号を
  `(m-1)*d_in+i`、列番号を`(n-1)*d_in+j`とする。
- `J[(m-1)*d_in+i, (n-1)*d_in+j] = Φ(|i><j|)[m,n]`。
  保存形式は非規格化Choi演算子で、TPなら`tr(J)=d_in`、`Tr_output(J)=I_input`。
  規格化されたChoi状態`J/d_in`は明示的な取得関数で返す。
- 内部伝播は`sum_i |i>_A|i>_S / sqrt(d)`という規格化状態を使用し、
  抽出時に`d`を掛けて公開Choi演算子へ変換する。
  TTの保存順`[ancilla, system, environment...]`と公開行列順を明示的に変換する。
- Juliaのcolumn-major `vec`規約で`vec(Φ(X))=S*vec(X)`となるsuperoperatorを提供する。
  reshapeだけに頼らず、上の添字定義に対応するreshufflingを実装・検証する。
- 入力演算子`X`の適用は`Φ(X)[m,n]=sum_ij J[m,i;n,j]*X[i,j]`。
  密度行列に限らず非Hermitianの行列単位にも対応する。

## コアAPIの責務

以下の名称は提案で、詳細なシグネチャは実装計画時に確定する。

- `ChoiMap`: 行列と入出力次元を保持する。形状と有限値を検証する。
  CP/TPをコンストラクタの必須条件にはしない。数値誤差や非CPの参照例も解析できる。
- `choi_matrix`、`choi_state`、`superoperator`、superoperatorからの逆変換。
- `apply(map, X)`とmapの合成。合成は独立したchannelの数学的操作であり、
  非Markov過程の異なる絶対時刻のmapをそのまま逐次適用する意味にはしない。
- `DynamicalMap`: 初期時刻、有限で非減少の時刻列、対応する`ChoiMap`の系列。
  入出力次元の整合性と長さを検証し、時刻補間や中間伝播を暗黙には行わない。
- 診断関数: Hermiticity残差、Choi固有値によるCP判定、output部分トレースの
  TP残差、Choi trace。許容誤差を引数で指定できる。
  CP判定はHermiticityを確認し、Hermitian部分の固有値だけで誤判定しない。
  結果へのCP/TP補正やtraceの再規格化は行わない。

## TTバックエンド

`ext/ChoiJamiolkowskiTTDynamicsExt.jl`でTTDynamicsとTTSolverを接続する。
存在しない`propagate_ksl`や`propagate_tamen`への依存は避ける。
現行TTDynamicsにはHEOM用の高水準伝播があるが、TFD用の同名APIはない。

1. `SBSystem`から既存の`tt_sbham`でHamiltonianと環境基底次元を構築する。
   `BosonicTFD`の周波数と結合係数は既に温度依存性を含む入力として扱う。
   温度や負周波数モードを二重に変換・追加しない。
2. `[ancilla, system, bath...]`順で、ancillaに恒等演算子を追加したTT Hamiltonianを作る。
   ユーザーが構築済みのTT Hamiltonianを渡す経路も用意し、単位は入力に従う。
3. 系–ancilla間のBell状態と、TFD変換後の環境真空をTTコアで直接構築する。
   高次元の全状態ベクトルや全Hamiltonianは生成しない。
4. 初期版の伝播器は論文と同じ対称KSL。TTSolverの`tt_ksl`は
   `dx/dt + A*x=0`を解くため、`A=im*H`を渡す。
   KSLは既存rankのままでは新たなentanglementを表現できないので、
   初期rankをHilbert空間で許される上限内で指定rankまで拡張する。
   真空のrank-1状態のまま伝播しない。rankと時間刻みは利用者が制御できる。
5. 各時刻で環境のdouble-layer transfer matrixを右から左へ縮約する。
   残ったancilla・系コアを縮約して`d^2 × d^2`のChoi行列だけdense化する。
   現行`TTDynamics.tt_partial_trace`は`tt_full`を呼ぶため、この経路には使わない。
6. 既定では時刻ごとのmapと診断を保存し、巨大なTT軌跡全体は保持しない。
   callbackで進捗やTT rank等を参照できるようにする。

状態norm保存だけでTPが保証されるとは主張しない。有限rank・時間刻みの影響は
ancilla周辺状態からTP残差として報告する。
初期版は時間に依存しないHamiltonianと、系–環境が初期に無相関である設定を対象とする。

## Process tensorへの拡張方針

入出力次元、脚順序、時刻、規格化、channel適用の規約を文書化して再利用可能にする。
将来のProcess tensorは各時刻のinput/output脚と介入操作の縮約を持つ別の型で実装する。
単一時刻のmap系列だけから一般の多時刻相関を復元できるとは扱わない。
未実装の多時刻APIや空のProcessTensor型を先回りして公開しない。

## 検証と例

- identity、複素unitary、振幅減衰channelを解析解と比較する。
- 複素の非Hermitian行列単位すべてに適用し、Choi・superoperator・map適用の
  添字と複素共役を検証する。非正方の入出力次元でも変換を確認する。
- TPでないmap、非CP map、非Hermitian Choi演算子の診断を確認する。
- 不均一次元・複素TT状態について、TT縮約と独立したdense部分トレースを比較する。
  本番縮約の経路が`tt_full`を必要としないことも確認する。
- 環境結合のある小規模spin-boson系で、初期Bell状態からKSL伝播したmapを
  denseの`exp(-im*H*t)`によるchannelと比較する。孤立系だけで検証を済ませない。
- パッケージ単体と、TTDynamics/TTSolverをロードした環境での両方を確認する。
- 手で指定した小さなTFD bathを持つ自己完結した実行例を同梱する。
  任意の初期密度行列へのmap適用と、CP/TP診断を示す。

## 実装上の制約と残る判断

Juliaは既存TTDynamicsと揃えて1.11以上。手元の実行環境は1.12.6。
TTDynamicsには多数の未コミット変更があるため、それらのファイルや結果データを変更せず、
新規パッケージを兄弟ディレクトリに作成する。
この設計書のみ、調査対象リポジトリの設計書置き場に追加している。
依存のローカル開発パスは利用者環境で解決し、公開パッケージに絶対パスを固定しない。
UUID、ファイル分割、内部ループとworkspace、細かいAPI名、テスト許容誤差、
ドキュメント言語の詳細は実装者の判断とする。

構成案への同意を受けて実装する。参照論文の公開年と依頼文の年の相違は上記に記録した。

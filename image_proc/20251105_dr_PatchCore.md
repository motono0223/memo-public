# **PatchCoreアルゴリズムの包括的分析：産業用異常検知におけるTotal Recallの実現**

## **序章：産業用異常検知におけるパラダイムシフト**

大規模な工業生産プロセスにおいて、欠陥部品を自動的に発見し、排除する能力は、品質保証とコスト管理の観点から極めて重要です 1。しかし、この分野の自動化は、長らく「コールドスタート」問題という根本的な課題に直面してきました 1。この問題は、トレーニング段階で利用可能なデータが、ほぼ「正常（欠陥なし）」なサンプルの画像のみに限られるという状況を指します 1。実際の製造ラインでは、異常なサンプル（欠陥品）は本質的に稀少であるか 3、あるいは未知のパターンで出現するため 4、従来の教師あり学習モデルの構築は事実上不可能です。したがって、このタスクは、正常データの分布から逸脱したサンプルを検出する「分布外（out-of-distribution）」検出の問題として定義されます 1。

このような背景のもと、Karsten Rothらによって発表された**PatchCore** 2 は、この分野におけるパラダイムシフトとなりました。このアルゴリズムは、「Towards Total Recall in Industrial Anomaly Detection（産業用異常検知における完全な再現率を目指して）」というタイトルの論文で提案され 1、コンピュータビジョンのトップカンファレンスであるCVPR 2022に採択されました 2。PatchCoreは、ImageNetのような大規模データセットで事前学習されたモデルの埋め込み（embeddings）と外れ値検出モデルを組み合わせる既存の研究ライン 1 を、革新的な方法で拡張するものです。

「Total Recall（完全な再現率）」という論文のタイトル 2 は、単なるキャッチフレーズではなく、産業応用におけるコストの非対称性を反映した、根本的な設計思想の転換を示しています。産業検査 8 において、異常検知の誤りには2種類あります。第一に、正常な部品を「異常」と誤判定する「偽陽性（False Positive）」です。第二に、異常な部品を「正常」と見逃す「偽陰性（False Negative）」です。偽陽性が引き起こすコストは主に検査員による再確認の手間ですが、偽陰性が引き起こすコスト（欠陥製品の市場流出、ブランド信頼の失墜、大規模リコール、安全上の問題）は、桁違いに甚大です。したがって、産業界の要求は「精度（Precision）」よりも「再現率（Recall）」、すなわち「欠陥を絶対に見逃さないこと」に強く偏っています。PatchCoreのアーキテクチャは、この「Total Recall」を達成するため、意図的に偽陰性を最小化する（＝高い感度を持つ）ように設計されています。

その結果は劇的であり、PatchCoreは、画像異常検知の主要なベンチマークであるMVTec AD 2 において、画像レベルの異常検出AUROC（Area Under the Receiver Operating Characteristic curve）スコアで最大99.6%という卓越した性能を達成しました 1。これは、当時の次点の競合手法と比較して、エラー率を半分以下に削減する成果です 1。

## **第1部：PatchCoreのアルゴリズム・アーキテクチャ詳解**

PatchCoreのアーキテクチャは、その高い性能にもかかわらず、驚くほどシンプルで直感的なコンポーネントで構成されています。その動作は、大きく「トレーニング」と「推論」の2つのフェーズに分けられます。

### **1.1. 思想的基盤：なぜパッチか？ なぜ中間特徴か？**

PatchCoreの核となるアイディアは、入力画像の局所的な「パッチ」の表現（patch representations）を異常検知の基礎として使用することです 4。この特徴抽出のプロセスには、ImageNetデータセットで事前学習されたResNet（Residual Network）ライクなディープ畳み込みニューラルネットワーク（CNN）が、特徴抽出器（feature extractor）として利用されます 4。

PatchCoreの設計において最も重要な決定の一つは、CNNのどの層から特徴を抽出するかという点です。論文の著者らは、ネットワークの「中間層（intermediate blocks）」の出力を意図的に選択し、最初（初期層）と最後（最終層）のブロックを無視します 4。例えば、広く使われているAnomalibライブラリの実装では、wide\_resnet50\_2バックボーンのlayers=\["layer2", "layer3"\]がデフォルトとして指定されています 8。

この「ゴルディロックス（Goldilocks）」とも呼べる特徴層の選定には、明確な理論的根拠があります。

1. **初期層（例：layer1）の棄却**：初期層は、「一般的すぎる（too generic）」特徴を捉えます 4。これらは、エッジ、コーナー、色といった、画像のごく低レベルな構成要素に対応します。しかし、産業用欠陥（例：正常な木目の中の「異常な」傷）は、低レベルな特徴（傷もまたエッジの集まり）だけでは、正常なパターンと区別することが困難です。  
2. **最終層（例：layer4, 5）の棄却**：対照的に、最終層は「ImageNetに偏りすぎている（too heavily based towards ImageNet）」と指摘されています 4。これらの層は、ImageNetの1,000クラス（犬、猫、車など）を分類するために、空間情報を破棄して高度に抽象化された「意味情報（セマンティクス）」に集約するように訓練されています。産業用欠陥は「犬」でも「猫」でもないため、この意味的な特徴空間では、「正常な金属表面」と「異常な傷」を区別することは、やはり困難になります。  
3. **中間層（例：layer2, 3）の採択**：その中間であるlayer2とlayer3こそが、異常検知に理想的な層です。これらは、低レベルな汎用性を超え、かつ高レベルな意味的抽象化には至らない、絶妙なバランスを保っています。これらの層は、空間情報を保持しつつ、「木目」「金属の光沢」「繊維の織り目」といった、局所的で複雑な*テクスチャ*や*パターン*を捉える能力に長けています。この「テクスチャルで非セマンティックな」特徴空間こそが、正常なテクスチャからのわずかな逸脱（＝異常）を検出するために最も適しているのです。

### **1.2. トレーニングフェーズ：正常性の「メモリバンク」構築**

PatchCoreのトレーニングフェーズは、正常（non-defective）なトレーニング画像のみを使用して、正常な状態とは何かを「記憶」するプロセスです。具体的には、収集された正常画像から、前述の中間層を用いてパッチ特徴を抽出し、それらをmemory\_bankと呼ばれる巨大な特徴リポジトリに保存（store）します 11。

ここで注目すべきは、PatchCoreが伝統的な深層学習の意味での「トレーニング」を必要としない点です。Anomalibライブラリのドキュメントが明確に指摘している通り、PatchCoreは「最適化/バックプロパゲーションを必要としません（requires no optimization/backpropagation）」 11。これは、事前学習済みのバックボーンの重みがトレーニング中に更新（ファインチューニング）されないこと、すなわち\*重みが固定（frozen）\*されていること 11 を意味します。

したがって、PatchCoreにおける「トレーニング」とは、実質的には「特徴抽出とインデックス作成」のワンパス（1エポック）のプロセスです 11。この設計は、実用上、計り知れない利点をもたらします。第一に、学習率の調整、エポック数の決定、収束の監視といった、時間のかかるハイパーパラメータ調整が不要になります。第二に、プロセスが決定論的であるため、再現性が極めて高くなります。そして第三に、「モデル」＝「データ（メモリバンク）」という、非常にシンプルで解釈しやすい構造が実現されます。

### **1.3. 推論フェーズ：逸脱の検出と局所化**

トレーニングフェーズで正常性の「辞書」としてのメモリバンクが構築されると、モデルは推論（テスト）の準備が整います。

パッチレベルのスコアリング  
テスト（推論）時、新しいテスト画像が入力されると、トレーニング時と全く同じプロセスでパッチ特徴が抽出されます 4。そして、抽出された各テストパッチ特徴 $m\_{test}$ は、トレーニング中に構築された広範なメモリバンク $M$ と比較されます 13。異常スコア $s^\*$ は、最も単純な形式では、$m\_{test}$ と、メモリバンク $M$ 内で最も近い特徴（最近傍）$m^\*$ との間の距離（例：ユークリッド距離）として計算されます 13。この計算は、 $s^\* \= ||m\_{test} \- m^\*||^2$ のように表されます 13。  
画像レベルのスコアリング  
単一のテスト画像は、多数のパッチから構成されます。画像全体の最終的な異常スコアは、これらの全パッチスコアの\*\*最大値（maximum distance）\*\*として決定されます 12。  
画像スコアとして「平均」や「中央値」ではなく「最大値」を選択するという設計は、序章で述べた「Total Recall」の哲学 2 と直接結びついています。このアルゴリズムの根底にある仮説は、「画像は、*単一でも*異常なパッチを含んでいれば、異常である」 12 というものです。例えば、ある画像が1,000個のパッチで構成され、そのうち999個が完全に正常であっても、たった1個のパッチ（例：微細な傷）が正常なメモリバンクからわずかに逸脱していれば、その（最大の）逸脱スコアが画像全体のスコアとして採用されます。結果として、その画像は「異常」とフラグ付けされます。これは、産業検査の現実（例：完全な製品に1つのネジが欠けているだけで欠陥品となる）を正確にモデル化した、非常に厳格で低寛容なポリシーです。

局所化（セグメンテーション）  
PatchCoreは、画像が異常かどうかを判定する（検出）だけでなく、異常が画像のどこに存在するかを特定（局所化）する能力も持ちます 4。これは、各パッチの異常スコア（メモリバンクへの最近傍距離）を計算し、そのスコアを元の画像上の対応するピクセル位置にマッピングし直すことで、「異常ヒートマップ（anomaly map）」を生成することによって達成されます 5。  
高度なスコアリングメカニズム  
多くの単純な説明では、異常スコアは「1つの最近傍への距離」とされています 12。しかし、PatchCoreの実際の設計、特にAnomalibのようなライブラリでの実装は、より高度なメカニズムの存在を示唆しています。AnomalibのPatchcoreモジュールは、num\_neighbors（近傍数）というパラメータを持ち、そのデフォルト値は9に設定されています 11。  
さらに、compute\_anomaly\_score関数 11 の説明によれば、この関数は単なるパッチスコアだけでなく、最近傍の「位置（locations）」と「埋め込み（embedding）」も引数に取り、「メモリバンク内の局所的な近傍構造（local neighborhood structure）を考慮する、論文の加重スコアリングメカニズム（weighted scoring mechanism）」を実装していると明記されています 11。

これは、PatchCoreのスコアリングが、単に「最も近い正常パッチからどれだけ離れているか？」だけでなく、「その最も近い正常パッチの*周辺*は、他の正常パッチでどれだけ密か？」も考慮していることを意味します。あるテストパッチが、正常な特徴空間の「まばら（sparse）」で孤立した領域に着地した場合、たとえ最近傍への距離が同じであっても、「密（dense）」なクラスタの中心付近に着地した場合よりも、高い異常スコアを受け取る可能性があります。このメカニズムにより、検出の頑健性が高められています。

## **第2部：中核技術：コアセット・サンプリングによる最適化**

PatchCoreのアーキテクチャは強力ですが、第1部で説明されたナイーブなアプローチ、すなわち「正常な全画像から抽出された*すべて*のパッチ特徴をメモリバンクに保存する」方法には、深刻なスケーラビリティの問題が内在します。

### **2.1. 課題：巨大なメモリバンクのジレンマ**

このアプローチの最大の問題は、メモリバンクのサイズです。メモリ使用量は、トレーニングデータセットのサイズ（画像枚数）に比例して指数関数的に増大します 15。高解像度の画像を多数使用する現実の産業アプリケーションでは、メモリバンクは容易に数十ギガバイトに達し、GPUメモリの容量を瞬く間に圧迫します 17。

さらに、この巨大なメモリバンクは、推論速度にも悪影響を及ぼします。高次元の特徴空間 16 において、入力されたすべてのテストパッチに対して最近傍探索を実行するプロセスは、膨大な計算オーバーヘッド 16 を生み出し、推論速度を著しく低下させます 16。この問題は、リソースが限られたエッジコンピューティングデバイス（例：工場のラインに設置されたスマートカメラ） 16 への展開を非現実的にします。

### **2.2. 解決策：k-Center Greedy (Minimax Facility Location) の適用**

PatchCoreは、このスケーラビリティのジレンマを解決するために、「コアセット・サブサンプリング（coreset subsampling）」 4 と呼ばれる強力な最適化技術を導入します。コアセットとは、元の巨大なデータセット（メモリバンク全体）の構造的特性を可能な限り保持した、小さな代表的なサブセット（部分集合）を指します 12。

PatchCoreで採用されている具体的なアルゴリズムは、「Minimax Facility Location Coreset Selection」 4、またはそれと密接に関連する「k-Center Greedy（k-中心貪欲法）」 11 と呼ばれます。

これらのアルゴリズムの目的は、元のメモリバンク $M$ 内の*すべて*の点 $m$ について、その点 $m$ からコアセット $C$ 内の*最も近い*点 $c^\*$ への距離を考え、これらの距離の*最大値*（ $\\max\_{m \\in M} ||m \- c^\*||$ ）を\*最小化（minimize）\*するようなコアセット $C$ を選択することです 4。

直感的に言えば、これは「正常な特徴空間全体を、指定された数（例：全体の1%）の『代表点』で*最も効率的にカバー*する」戦略です。「Greedy（貪欲）」アルゴリズム 19 は、このカバレッジを効率的に達成します。まずランダムに1つの点をコアセットに追加し、次に「現在コアセットに選択されているどの代表点からも*最も遠い*位置にある」メモリバンク内の点を、代表点として繰り返し追加していきます。このプロセスにより、代表点は自動的に、まだカバーされていない特徴空間の「隙間」を埋めるように配置されていきます。

### **2.3. 実証的効果とトレードオフ**

実証的効果  
この最適化の効果は劇的です。原論文 4 および実例 5 によれば、元のメモリバンクに保存されている全パッチ表現のわずか1% 4（あるいは10% 8）をサンプリングしてコアセットを構築するだけで、SOTA（最高水準）のパフォーマンスをほぼ維持したまま、推論時間を大幅に短縮（例：200ms未満）できることが示されています 4。これにより、PatchCoreは高い精度と高速な推論を両立することが可能になりました。  
トレードオフ (1)：推論速度 vs コアセット構築速度  
コアセット・サンプリングは、推論時（Inference）の最近傍探索を劇的に高速化します 4。しかし、このコアセット自体を構築するプロセス（「トレーニング」フェーズの最終ステップ）は、それ自体が計算コストのかかるステップであるというトレードオフが存在します。  
この点は、GitHub上のIssue（問題報告） 21 で明確に示されています。あるユーザーは、「k-Center GreedyによるCoreSet Samplerの作成」ステージで処理が「スタック」し、CPU使用率が高騰していると報告しています。これは、PatchCoreの「トレーニング」コストがゼロではないことを示しています。バックプロパゲーションのコストはかかりませんが、代わりに、(a) すべての正常画像に対するフォワードパス（特徴抽出）、そして (b) 抽出された*すべて*の特徴に対する、計算量の多いGreedyなクラスタリング（コアセット構築）という、2段階のコストが発生します。

IntelのAnomalibライブラリのv2.2.0 22 が、改善点として「PatchCoreモデルのより高速なコアセット選択（\~30%高速なトレーニング）」を挙げていることからも、これが現実的なボトルネックであることが裏付けられます。したがって、オンデバイスでの再トレーニングなど、展開シナリオを検討する上で、この「推論は速いが、コアセット構築には時間がかかる」という特性の理解が不可欠です。

トレードオフ (2)：Greedyサンプリングの脆弱性  
k-Center Greedy戦略は効率的ですが、潜在的な脆弱性も抱えています。ある研究 23 は、「PatchCoreはコアセットサンプリングにGreedy戦略を使用しており、これは特徴空間の外れ値（outliers）を選好する（favors）」と指摘しています。  
これは、k-Center Greedyのアルゴリズム（「最も遠い点を選ぶ」 19）の性質に起因します。もしトレーニング用の「正常」データセットに、ノイズや、ラベル付けされていない微細な異常が（誤って）混入していた場合、アルゴリズムはこれらの「外れ値」を「代表点」として積極的に選択してしまう危険性があります。これにより、異常なサンプルが「正常」のコアセット・メモリバンクに組み込まれ、結果としてモデルの性能が（23によれば最大40% AUROC低下という）「壊滅的に」悪化する可能性があります。これは、クリーンなベンチマークデータ（MVTecなど）から、ノイズの多い実世界の製造ラインデータに移行する際の、最大のリスク要因の一つと言えます。

## **第3部：パフォーマンス分析とベンチマーキング**

PatchCoreの有効性は、特にMVTec AD（Anomaly Detection）ベンチマークにおいて、既存の手法と比較して圧倒的な結果を示したことで確立されました。

### **3.1. MVTec ADにおける実証結果**

MVTec ADは、産業用異常検知の分野で広く使用されている標準ベンチマークであり、15種類の異なるカテゴリ（繊維、金属、木材など）の画像を含みます 13。PatchCoreは、この挑戦的なデータセットにおいて、以下のSOTA性能を達成しました。

* **画像レベル（検出）**：画像全体が正常か異常かを判定するタスクにおいて、AUROCスコアで最大\*\*99.6%\*\*というほぼ完璧なスコアを記録しました 1。これは、次点の競合手法（PaDiM）と比較してエラー率を半分以下に削減することを意味します 1。  
* **ピクセルレベル（局所化）**：異常な領域をピクセル単位で特定するタスクにおいても、ピクセルレベルAUROC（Seg. AUROC）で平均98.1% \- 98.4% 8、PROスコア（Per-Region Overlap score）で93.5% \- 95%超 8 という高い性能を示し、欠陥領域の正確な特定能力を実証しました。

以下のテーブルは、主要な競合手法であるSPADEおよびPaDiMと、PatchCoreの性能を主要なメトリクスで比較したものです。

**テーブル 1: PatchCore vs 競合手法 (MVTec AD ベンチマーク)**

| 手法 | バックボーン | 画像レベル AUROC (%) | ピクセルレベル AUROC (%) | PROスコア (%) | 推論時間 (s) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| PaDiM 24 | Wide-ResNet50 | 95.4 \- 97.9 | 97.3 \- 97.5 | 91.8 \- 92.1 | 0.19 |
| **PatchCore (1% Coreset)** 24 | Wide-ResNet50 | **99.0 \- 99.2** | **98.0 \- 98.1** | **93.1 \- 93.5** | **0.17** |
| **PatchCore (Ensemble)** 8 | Ensemble | **99.6** | **98.2** | **94.9** | (N/A) |

注：PaDiMおよびSPADEの数値は、PatchCore論文 24 内での再実装・比較に基づいています。推論時間は環境に依存します。

### **3.2. 競合手法との構造的比較：PatchCore vs. PaDiM**

上記のテーブルは、PatchCoreがPaDiMやSPADEといった既存のSOTA手法を性能面で凌駕していることを明確に示しています。この優位性は、単なる数値的な改善ではなく、アルゴリズムの根本的な設計思想の違いに起因しています 24。

PaDiM（およびSPADE 26）の決定的な弱点は、その*位置依存性*にあります。PaDiMは、テスト画像の特定のパッチ（例：画像の左上隅）を評価する際、トレーニングデータセットの*同じ空間的位置*（例：すべての正常画像の左上隅）から取得されたパッチの特徴と比較します 24。

対照的に、PatchCoreは、この空間的な制約を完全に排除しました。PatchCoreでは、テストパッチは、トレーニング画像の*すべての位置*から収集された「グローバルな」メモリバンク全体と比較されます 24。この設計上の違いが、いくつかの決定的な利点をもたらします。

1. **位置合わせへの非依存**：グローバルなメモリバンクを使用することで、PatchCoreはPaDiMよりも「画像の位置合わせへの依存度が低い（less reliant on image alignment）」 24 です。PaDiMは、検査対象物が常にカメラに対して完璧に同じ位置・同じ向きで配置されていることを暗黙的に仮定しますが、PatchCoreは多少の位置ずれや回転に対して、より柔軟に対応できます。  
2. **より大きな文脈の利用**：PatchCoreは「はるかに大きな公称コンテキスト（much larger nominal context）」 24 を利用して異常を推定します。PaDiMが「左上隅」の正常性しか知らないのに対し、PatchCoreは「画像全体のすべての正常なテクスチャ」を知っており、よりロバストな判定が可能になります。  
3. **柔軟な解像度**：PaDiMとは異なり、PatchCoreはトレーニング時とテスト時で入力画像の形状（解像度）が同じである必要がありません 24。

### **3.3. パフォーマンスと速度の両立**

ベンチマーキング（テーブル1）における最大の成果は、PatchCore-1%（コアセット）が、PaDiMよりも**高速な推論時間**（0.17秒 vs 0.19秒）で、PaDiMよりも**遥かに高い精度**（画像レベルAUROC 99.0% vs 95.4%）を同時に達成した点です 24。

これは、コアセット・サンプリングが、パフォーマンスを犠牲にすることなく、むしろ向上させながら、速度のボトルネックを解消する、理想的な最適化であることを示しています。PatchCoreは、産業界が求める「高精度」と「リアルタイム性」という二律背反の要求を、見事に両立させたのです。

## **第4部：実世界への展開：応用事例と産業実装**

PatchCoreが学術的なベンチマークで示した卓越した性能は、即座に産業界と関連分野の研究者の注目を集め、現実世界のアプリケーションへと急速に展開されています。

### **4.1. 産業用自動検査（AVI）**

PatchCoreの主なターゲット領域は、産業用ビジョン（Industrial Vision）を用いた自動外観検査（Automated Visual Inspection, AVI）です 1。

具体的なユースケースとしては、プリント基板（PCB）の微細なはんだ付け不良や断線の検出 5、高速鉄道の部品（例：ボルト、ベアリング）の亀裂検査 13、収穫されたリンゴの傷や腐敗の検出 29 などが挙げられます。その他、MVTec ADデータセット 8 に含まれるような、繊維のほつれ、金属部品の傷、木材の節穴など、多種多様な製造品目の検査に適用されています。

2022年のCVPRでの発表からわずかな期間で、PatchCoreが主要な商用・オープンソースのMLOps（機械学習基盤）およびEdgeAIプラットフォームに標準機能として搭載されたことは、その実用性の強力な証拠です。

* **Edge Impulse**（主要なエッジAIプラットフォーム）は、エンタープライズユーザー向けにPatchCoreを「Visual Anomaly Detection (FOMO-AD) ラーニングブロック」に統合したことを発表しています 30。  
* **Clarifai**（AIプラットフォーム）は、PatchCoreを「SOTA（最高水準）の視覚的異常検知手法」としてブログで特集し、自社ソリューションへの組み込みを示唆しています 3。  
* **MathWorks**（MATLABの開発元）は、PatchCoreを「Automated Visual Inspection Library for Computer Vision Toolbox」の一部として提供し、PCB欠陥検出の具体的なチュートリアル 5 を公開しています。  
* **Intel** は、オープンソースのAnomalibライブラリ 4 を通じて、PatchCoreの最適化された（22）実装を主導し、コミュニティの発展に貢献しています。

この異例の速さでの商用採用は、PatchCoreが、産業界が長年求めていた3つの重要な要件、すなわち (a) 欠陥を見逃さない高い精度、(b) 製造ラインのタクトタイムに対応できる高速な推論、(c) 正常品データだけで学習できる「コールドスタート」問題への完全な適合 3、を完璧に満たしていたことを示しています。

### **4.2. 医療画像診断への応用**

PatchCoreの応用範囲は、管理された環境の多い産業分野にとどまらず、より複雑で多様性の高い医療画像 4 の分野にも拡大しています。特に、胸部X線写真（CXR）における病変の検出 4 など、異常が稀で多様な形態をとる医療診断支援への適用が研究されています。

しかし、PatchCoreを製造ライン（均一な背景、制御された照明）から医療画像（複雑な解剖学的構造、ノイズ、個体差）に移行させることは、単純な「プラグアンドプレイ」ではありません。2024年に発表された研究 32 は、CXR画像にPatchCoreを直接適用する際の2つの主要な課題を明らかにしています。第一に、病変が特定の*臓器*（例：肺野）にのみ発生するというドメイン知識の組み込み。第二に、病変とは無関係な画像*ノイズ*（例：撮影機器のアーチファクト）の存在です。

これらの課題に対処するため、研究者たちはPatchCoreの「前段」と「後段」に、医療画像ドメイン固有のモジュールを追加する必要がありました 32。

1. **前処理（Pre-processing）**：アフィン変換（Affine Transformation）を用いた「画像アライメント（位置合わせ）」を導入し、撮影されたすべてのX線画像を標準的な構図に自動的に揃えます 32。  
2. **後処理（Post-processing）**：「特徴マップのハードマスキング（Feature map hard masking）」を導入し、関心のある解剖学的領域（例：肺野）以外の領域から抽出された特徴を計算から除外します 32。

この事例は、PatchCoreが強力な「特徴比較エンジン」である一方で、産業分野のような構造化されたタスク以外（非構造化データ）に適用する際は、ドメイン知識に基づいた強力な前処理・後処理パイプラインと組み合わせることで、初めてその真価を発揮することを示唆しています。

## **第5部：批判的評価とアルゴリズムの進化**

PatchCoreはSOTAを達成しましたが、万能ではありません。その成功と普及は、同時にアルゴリズム固有の制限や限界も明らかにしました。現在、研究コミュニティはこれらの課題を克服するため、PatchCoreを基盤とした新たな派生型（バリアント）の研究を活発に進めています。

### **5.1. PatchCoreの固有の制限**

複数の研究 16 が、PatchCoreの主要な制限を指摘しています。

(1) 位置変動への脆弱性  
これがPatchCoreの最大の弱点です。元のPatchCoreは、サンプル間に回転（rotation）、反転（flipping）、またはわずかな位置ずれ（misaligned pixels）といった「位置関係（positional relationships）」が存在する場合、「異常を正確に特定する上で重大な制限（significant limitations）」に直面します 33。  
この位置変動への脆弱性は、異方性（例：一方向の木目や繊維）を持つ複雑なテクスチャ 35 の検査において、特に問題となります。その根本的な原因は、バックボーン（WideResNet50）の中間層特徴の性質にあります。layer2やlayer3から抽出される特徴は、畳み込みの性質上、ある程度の「並進（translation）」不変性（正しくは同変性）は持ちますが、「回転（rotation）」不変性は持ちません。

その結果、PatchCoreのメモリバンクは「上向きの正常な木目」は正常として記憶しますが、「90度回転した正常な木目」は未知のパターン（＝異常）として誤検出してしまいます 35。これは、33が指摘する「回転」や「反転」の問題と根本的に同じ課題です。

(2) メモリと計算のオーバーヘッド  
第2部で詳述した通り、高次元のパッチ特徴 16 と、データセットサイズに比例してスケールするメモリバンク 15 は、深刻な計算オーバーヘッドとなります。この問題は、特にリソースが限られたエッジデバイス 16 への展開を妨げる最大の要因の一つです。2024年から2025年にかけての最新の研究 16 が、依然としてこのメモリ効率の問題に焦点を当てていることからも、これが現在進行形の重要な課題であることがわかります。  
(3) コアセットの脆弱性  
第2部で指摘した通り、k-Center Greedyによるコアセット選択戦略は、トレーニングデータ内の「外れ値（ノイズや未ラベルの異常）」を選好し 23、メモリバンクを汚染するリスクがあります。

### **5.2. 派生型（バリアント）による課題克服**

PatchCoreの派生型の研究動向は、上記（5.1）で特定された弱点に直接対応する形で進化しています。これは、PatchCoreが「拡張可能なプラットフォーム」として機能していることを示しています。

* **弱点：位置変動への脆弱性** 33  
  * → 解決策：FR-PatchCore 13  
    この派生型は、Spatial Transformer Network (STN) をアーキテクチャに導入し、特徴を比較する前に「特徴レベルのレジストレーション（位置合わせ）」を学習させることで、この問題を正面から解決しようと試みています 33。  
* **弱点：少数サンプル（Few-Shot）での不安定性** 2  
  * → 解決策：Optimizing PatchCore (Few-Shot PatchCore) 36  
    この研究では、「アンチエイリアス処理（anti-aliased）」を施した、より制約の強い（＝並進不変性が高い）特徴抽出器を使用します。これにより、より強力な帰納バイアスが働き、少数のデータからでも一般化可能な特徴の学習が可能になると期待されます 36。  
* **弱点：バックボーンと特徴集約の複雑さ** 38  
  * → 解決策：AnomalyDINO 38  
    このアプローチは、バックボーンをResNetから自己教師あり学習モデルであるDINOv2に置き換えることで、特徴エンジニアリング（どの層をどう組み合わせるか）の段階を簡素化します。  
* **弱点：メモリと計算のオーバーヘッド** 16  
  * → 解決策：Memory-Efficient PatchCore 16  
    2024-2025年の研究 16 では、計算コストの高いk-Center Greedyの代わりに、PCA（主成分分析）による次元削減と、より軽量なK-Meansクラスタリングを統合し、メモリフットプリントを削減するアプローチが探求されています。  
* **弱点：パッチ間の文脈（コンテキスト）の無視** 40  
  * → 解決策：SA-PatchCore 40  
    自己注意（Self-Attention）モジュールを導入し、空間的に離れたパッチ間の「共起関係（co-occurrence relationships）」の異常（例：「ネジ」の隣には必ず「ワッシャー」があるはず、など）も検出します。  
* **弱点：既知の異常データを活用できない** 41  
  * → 解決策：Labeled PatchCore 41  
    「正常」のコアセットに加え、利用可能な少数の異常データから「異常」のコアセット（anomaly coreset）も作成し、分類性能を向上させます。

## **第6部：実装エコシステムと総括**

PatchCoreの理論的な優位性だけでなく、その利用しやすさが、アルゴリズムの急速な普及を支えています。

### **6.1. 実装ガイド：リポジトリとライブラリ**

PatchCoreを利用しようとする開発者や研究者には、主に2つの信頼できる実装ソースが存在します。

Amazon Science 公式リポジトリ (patchcore-inspection)  
原論文の著者らによる公式実装であり 1、アルゴリズムの挙動を最も忠実に再現しています。トレーニングと評価は、主にコマンドラインインターフェース（CLI）を通じて実行されます。

* **トレーニング**：bin/run\_patchcore.py スクリプトを使用します 8。  
* **評価**：bin/load\_and\_evaluate\_patchcore.py スクリプトを使用します 8。

主要なCLIパラメータ 8 には以下が含まれます。

* \-b (backbone)：使用するバックボーン（例: wideresnet50）。  
* \-le (layer)：特徴抽出層（例: layer2, layer3）。  
* sampler：サンプリングパラメータのセクションを開始。  
* \-p (percentage)：コアセットのサンプリング率（例: 0.1 \= 10%）。  
* approx\_greedy\_coreset：使用するサンプラー（k-Center Greedy）。  
* \--faiss\_on\_gpu：GPU上での高速な類似性検索の実行。

ここで、--faiss\_on\_gpu 8 というオプションは、単なる実装の詳細ではなく、PatchCoreの速度最適化における第二の柱です。PatchCoreの高速な推論は、(1) コアセット・サンプリング（探索対象の*点の数*を減らす）と、(2) FAISS（Facebook AI Similarity Search）ライブラリ（探索プロセス*自体*を高速化する）という、2つの技術の組み合わせによって実現されています。

Intel Anomalib ライブラリ  
Intelによって開発されたAnomalib 4 は、異常検知アルゴリズムの包括的なベンチマーク・ライブラリであり、PyTorch Lightningに基づいて構築されています 14。PatchCoreのモジュール化されたクリーンな実装 11 を提供しており、多くの実務家にとってのデファクトスタンダードとなっています。  
Anomalibは、PatchCoreの利用を大幅に簡素化します。42のGitHubディスカッションでは、あるユーザーが「正常データのみ」という現実的なシナリオでモデルをトレーニングしようとしてエラーに直面している様子が描かれており、理論（論文）と実践（現実のデータパイプライン）の間のギャップを示しています。Anomalibは、backbone, layers, coreset\_sampling\_ratioといった主要なパラメータを整理し、Lightningのフレームワークで抽象化することで、このギャップを埋め、研究者ではない開発者でもSOTAアルゴリズムを容易に利用できるようにする、重要な「橋渡し」の役割を果たしています 11。

さらに、AnomalibはIntelによって活発にメンテナンスされており、22が示すように、v2.2.0ではPatchCoreの主要なボトルネックであった「コアセット選択の高速化（\~30%）」と「メモリ使用量の削減」が図られるなど、コミュニティによる継続的な最適化の恩恵を受けられます。

**テーブル 2: AnomalibにおけるPatchCoreの主要設定パラメータ (v2.0.0)**

| パラメータ名 | 説明 | デフォルト値 | 専門家による調整TIPS |
| :---- | :---- | :---- | :---- |
| backbone | 特徴抽出に使用するCNNバックボーン。 | "wide\_resnet50\_2" | WR50は性能と速度のバランスが良い。より重いモデル（WR101等）は精度が上がる可能性があるが、推論は遅くなる。 |
| layers | 特徴を抽出するバックボーンの層。 | ("layer2", "layer3") | この組み合わせが「ゴルディロックス」層であり、SOTA性能の鍵（1.1節参照）。layer1（一般的すぎ）やlayer4（特化しすぎ）の追加は通常推奨されない。 |
| pre\_trained | バックボーンにImageNet事前学習重みを使用するか。 | True | **Trueから変更すべきではない。** PatchCoreの性能は、ImageNetで学習された強力な汎化特徴表現に完全に依存している。 |
| coreset\_sampling\_ratio | メモリバンクをサブサンプリングする割合。 | 0.1 (10%) | **最も重要なトレードオフ用パラメータ。** 0.1でも高性能 8。1%（0.01）まで下げても高性能を維持できる場合がある 4。値を*下げる*と、メモリ使用量と推論時間は*減少*するが、精度が低下する可能性がある。値を*上げる*（例: 1.0、サンプリングなし）と、精度は最大になるが、メモリと推論時間が爆発する 15。 |
| num\_neighbors | 異常スコア計算に使用する最近傍の数。 | 9 | （1.3節参照）1（単純な最大距離）よりも大きな値に設定することで、メモリバンクの局所的な密度を考慮した、より頑健なスコアリングが可能になる 11。 |

### **6.2. 結論と将来展望**

総括  
PatchCore（CVPR 2022）は、(1) 強力な事前学習済み中間特徴、(2) 巨大な正常メモリバンク、(3) k-Center Greedyによる効率的なコアセット・サンプリング、という3つの要素を独創的に組み合わせることで、産業用異常検知における「コールドスタート」問題を解決するSOTAソリューションを確立しました。その「Total Recall」を目指す哲学は、極めて高い検出性能と高速な推論速度を両立させ、学術界と産業界の両方で即座に受け入れられる成果となりました。  
将来展望  
PatchCoreの成功は、同時にその限界（位置変動への脆弱性、メモリのオーバーヘッド、コアセットの脆弱性）も明らかにしました。現在の研究の最前線は、PatchCoreを「プラットフォーム」として利用し、これらの弱点をモジュラー的に克服することに焦点を当てています（FR-PatchCore, Memory-Efficient PatchCoreなど） 16。  
今後の展望として、特に、(a) エッジデバイスへの広範な展開を可能にするための、PCAやK-Meansを用いたメモリ効率のさらなる追求 16、および (b) アンチエイリアシングやSTNの導入による、特徴空間自体の不変性（回転など）の向上 33 が、PatchCoreの適応範囲を、管理された工場環境から、より複雑な非産業シーン（医療、監視、自動運転など）へと拡大していくための鍵となるでしょう。

#### **引用文献**

1. Towards Total Recall in Industrial Anomaly Detection \- CVF Open Access, 11月 5, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2022/papers/Roth\_Towards\_Total\_Recall\_in\_Industrial\_Anomaly\_Detection\_CVPR\_2022\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)  
2. \[2106.08265\] Towards Total Recall in Industrial Anomaly Detection \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/abs/2106.08265](https://arxiv.org/abs/2106.08265)  
3. Visual Anomaly Detection for Manufacturing: From SOTA Model to Clarifai Solutions, 11月 5, 2025にアクセス、 [https://www.clarifai.com/blog/visual-anomaly-detection-for-manufacturing-from-sota-model-to-clarifais-solutions](https://www.clarifai.com/blog/visual-anomaly-detection-for-manufacturing-from-sota-model-to-clarifais-solutions)  
4. Anomaly detection in images using PatchCore \- dataroots, 11月 5, 2025にアクセス、 [https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore](https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore)  
5. Localize Industrial Defects Using PatchCore Anomaly Detector \- MATLAB & Simulink, 11月 5, 2025にアクセス、 [https://www.mathworks.com/help/vision/ug/detect-pcb-defects-using-patchcore-detector.html](https://www.mathworks.com/help/vision/ug/detect-pcb-defects-using-patchcore-detector.html)  
6. Towards Total Recall in Industrial Anomaly Detection \- Semantic Scholar, 11月 5, 2025にアクセス、 [https://www.semanticscholar.org/paper/Towards-Total-Recall-in-Industrial-Anomaly-Roth-Pemula/23ad8fc48530ce366f8192dfb48d0f7df1dba277](https://www.semanticscholar.org/paper/Towards-Total-Recall-in-Industrial-Anomaly-Roth-Pemula/23ad8fc48530ce366f8192dfb48d0f7df1dba277)  
7. Towards Total Recall in Industrial Anomaly Detection | Empirical Inference \- Max Planck Institute for Intelligent Systems, 11月 5, 2025にアクセス、 [https://is.mpg.de/ei/publications/rothetal22](https://is.mpg.de/ei/publications/rothetal22)  
8. amazon-science/patchcore-inspection \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)  
9. Towards Total Recall in Industrial Anomaly Detection \- ResearchGate, 11月 5, 2025にアクセス、 [https://www.researchgate.net/publication/363910462\_Towards\_Total\_Recall\_in\_Industrial\_Anomaly\_Detection](https://www.researchgate.net/publication/363910462_Towards_Total_Recall_in_Industrial_Anomaly_Detection)  
10. Fast Anomaly Detection for Vision-Based Industrial Inspection Using Cascades of Null Subspace PCA Detectors \- PubMed Central, 11月 5, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12349016/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349016/)  
11. PatchCore — Anomalib documentation, 11月 5, 2025にアクセス、 [https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html)  
12. PatchCore \- Anomalib v0.3.7, 11月 5, 2025にアクセス、 [https://anomalib.readthedocs.io/en/v0.3.7/reference\_guide/algorithms/patchcore.html](https://anomalib.readthedocs.io/en/v0.3.7/reference_guide/algorithms/patchcore.html)  
13. FR-PatchCore: An Industrial Anomaly Detection Method for Improving Generalization \- MDPI, 11月 5, 2025にアクセス、 [https://www.mdpi.com/1424-8220/24/5/1368](https://www.mdpi.com/1424-8220/24/5/1368)  
14. Anomalib: Image Anomaly Detection with PatchCore \- Kaggle, 11月 5, 2025にアクセス、 [https://www.kaggle.com/code/raffelsbem98/anomalib-image-anomaly-detection-with-patchcore](https://www.kaggle.com/code/raffelsbem98/anomalib-image-anomaly-detection-with-patchcore)  
15. Patchcore fails to train with a large dataset · open-edge-platform anomalib · Discussion \#802, 11月 5, 2025にアクセス、 [https://github.com/openvinotoolkit/anomalib/discussions/802](https://github.com/openvinotoolkit/anomalib/discussions/802)  
16. Efficiency-Optimized PatchCore for Anomaly Detection in Future Edge Deployment \- Webthesis \- Politecnico di Torino, 11月 5, 2025にアクセス、 [https://webthesis.biblio.polito.it/35452/1/tesi.pdf](https://webthesis.biblio.polito.it/35452/1/tesi.pdf)  
17. Divide and Conquer: High-Resolution Industrial Anomaly Detection via Memory Efficient Tiled Ensemble \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2403.04932v1](https://arxiv.org/html/2403.04932v1)  
18. PaSTe: Improving the Efficiency of Visual Anomaly Detection at the Edge \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2410.11591v1](https://arxiv.org/html/2410.11591v1)  
19. anomalib.models.components.sampling.k\_center\_greedy, 11月 5, 2025にアクセス、 [https://anomalib.readthedocs.io/en/v0.3.6/api/anomalib/models/components/sampling/k\_center\_greedy/index.html](https://anomalib.readthedocs.io/en/v0.3.6/api/anomalib/models/components/sampling/k_center_greedy/index.html)  
20. Implementation of anomaly detection method PatchCore. \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/ComputermindCorp/patchcore](https://github.com/ComputermindCorp/patchcore)  
21. Issue in Patch-Core · Issue \#236 · open-edge-platform/anomalib \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/openvinotoolkit/anomalib/issues/236](https://github.com/openvinotoolkit/anomalib/issues/236)  
22. open-edge-platform/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference. \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)  
23. SoftPatch: Unsupervised Anomaly Detection with Noisy Data \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2403.14233v1](https://arxiv.org/html/2403.14233v1)  
24. \[2106.08265\] Towards Total Recall in Industrial Anomaly Detection, 11月 5, 2025にアクセス、 [https://ar5iv.labs.arxiv.org/html/2106.08265](https://ar5iv.labs.arxiv.org/html/2106.08265)  
25. Towards Total Recall in Industrial Anomaly Detection | alphaXiv, 11月 5, 2025にアクセス、 [https://www.alphaxiv.org/overview/2106.08265v2](https://www.alphaxiv.org/overview/2106.08265v2)  
26. Tailored Transformation Invariance for Industrial Anomaly Detection \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2509.17670v1](https://arxiv.org/html/2509.17670v1)  
27. AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2408.15113v1](https://arxiv.org/html/2408.15113v1)  
28. Localize Industrial Defects Using PatchCore Anomaly Detector \- MATLAB & Simulink, 11月 5, 2025にアクセス、 [https://la.mathworks.com/help/vision/ug/detect-pcb-defects-using-patchcore-detector.html](https://la.mathworks.com/help/vision/ug/detect-pcb-defects-using-patchcore-detector.html)  
29. Leveraging Unsupervised Learning for Cost-Effective Visual Anomaly Detection \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2409.15980v1](https://arxiv.org/html/2409.15980v1)  
30. PatchCore Boosts Visual Anomaly Detection in Edge Impulse, 11月 5, 2025にアクセス、 [https://www.edgeimpulse.com/blog/patchcore-boosts-visual-anomaly-detection-in-edge-impulse/](https://www.edgeimpulse.com/blog/patchcore-boosts-visual-anomaly-detection-in-edge-impulse/)  
31. Anomalib Documentation — Anomalib documentation, 11月 5, 2025にアクセス、 [https://anomalib.readthedocs.io/](https://anomalib.readthedocs.io/)  
32. Region and Global-Specific PatchCore based Anomaly Detection ..., 11月 5, 2025にアクセス、 [https://koreascience.kr/article/JAKO202427443329031.page](https://koreascience.kr/article/JAKO202427443329031.page)  
33. FR-PatchCore: An Industrial Anomaly Detection Method for ... \- NIH, 11月 5, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10934034/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934034/)  
34. FR-PatchCore: An Industrial Anomaly Detection Method for Improving Generalization, 11月 5, 2025にアクセス、 [https://www.researchgate.net/publication/378351078\_FR-PatchCore\_An\_Industrial\_Anomaly\_Detection\_Method\_for\_Improving\_Generalization](https://www.researchgate.net/publication/378351078_FR-PatchCore_An_Industrial_Anomaly_Detection_Method_for_Improving_Generalization)  
35. Zero-shot Texture Anomaly Detection \- Tohoku CVLab, 11月 5, 2025にアクセス、 [https://www.vision.is.tohoku.ac.jp/?p=383\&lang=en](https://www.vision.is.tohoku.ac.jp/?p=383&lang=en)  
36. Implementation of our paper "Optimizing PatchCore for Few/many-shot Anomaly Detection" \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/scortexio/patchcore-few-shot](https://github.com/scortexio/patchcore-few-shot)  
37. Optimizing PatchCore for Few/many-shot Anomaly Detection, 11月 5, 2025にアクセス、 [https://arxiv.org/abs/2307.10792](https://arxiv.org/abs/2307.10792)  
38. AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2405.14529v2](https://arxiv.org/html/2405.14529v2)  
39. AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2 \- CVF Open Access, 11月 5, 2025にアクセス、 [https://openaccess.thecvf.com/content/WACV2025/papers/Damm\_AnomalyDINO\_Boosting\_Patch-Based\_Few-Shot\_Anomaly\_Detection\_with\_DINOv2\_WACV\_2025\_paper.pdf](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf)  
40. (PDF) SA-PatchCore: Anomaly Detection in Dataset With Co-Occurrence Relationships Using Self-Attention \- ResearchGate, 11月 5, 2025にアクセス、 [https://www.researchgate.net/publication/366910335\_SA-PatchCore\_Anomaly\_Detection\_in\_Dataset\_with\_Co-occurrence\_Relationships\_Using\_Self-attention](https://www.researchgate.net/publication/366910335_SA-PatchCore_Anomaly_Detection_in_Dataset_with_Co-occurrence_Relationships_Using_Self-attention)  
41. Domain-independent detection of known anomalies \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2407.02910v1](https://arxiv.org/html/2407.02910v1)  
42. PatchCore: Training only on good images and then inference on good and bad images · open-edge-platform anomalib · Discussion \#192 \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/open-edge-platform/anomalib/discussions/192](https://github.com/open-edge-platform/anomalib/discussions/192)

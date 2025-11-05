# **AI生成異常画像に対する自動アノテーション戦略：Visual Grounding基盤モデルと教師なし異常検出の比較分析**

## **序章：課題と本レポートの構成**

### **1.1. ユーザーの目的と現在のワークフロー分析**

ユーザーは、異常検出データセットの作成を目的としています。現在、(1) 大量に保有する正常画像を活用し、(2) Azure gpt-image-1 などの生成AIを用い、(3) 「言語（テキストプロンプト）」で異常ケースを指定して異常画像を生成する、というワークフローを採用しています。

当面の課題は、この生成された異常画像に対し、「どこが異常であるか」を効率的に、かつ自動でアノテーション（Bounding Boxまたはセグメンテーションマスク）することです。このため、テキストプロンプトに基づいて画像内の特定領域を自動でローカライズする技術が求められています。

### **1.2. 本レポートが提示する2つの戦略**

この課題に対し、本レポートは2つの異なる戦略（Strategy）を提示します。

1. 戦略A：プロンプト・トゥ・アノテーション（第1部・第2部）  
   これは、ユーザーの現在の要求に直接応えるアプローチです。gpt-image-1 で使用した「言語プロンプト」（例：「"ケーブルの被覆に生じた亀裂"」）を、最新の視覚言語基盤モデル（VLM）に入力し、該当領域の座標（Box）やマスク（Mask）を直接出力させます。これは「Visual Grounding（ビジュアル・グラウンディング）」または「Referring Expression Segmentation（指示表現セグメンテーション）」として知られるタスクです 1。  
2. 戦略B：教師なし異常検出（第3部）  
   これは、より根本的なパラダイム転換を提案するアプローチです。ユーザーが「正常画像をたくさんある」と明記している点に着目し、異常画像の生成やアノテーションを一切行わず、正常画像のみを学習して異常を検出する「教師なし異常検出（Unsupervised Anomaly Detection: UAD）」モデルを構築します 5。このアプローチは、ユーザーの最終目的（異常検出器の構築）への最短経路である可能性が高く、同時に、戦略Aのための強力な自動アノテーションツールとしても機能します。

本レポートでは、これら2つの戦略を詳細に分析し、それぞれに最適な基盤モデル、クラウドサービス、および具体的な実装手順を提示します。

---

## **第1部：戦略A（自己ホスト型）：プロンプトによる自動アノテーション基盤モデル**

このセクションでは、ユーザーが自身でホスト・実行可能なオープンソースの基盤モデル（VLM）を分析し、テキストプロンプトからBounding Boxまたはセグメンテーションマスクを生成する手法を解説します。

### **1.1. 中核概念：Visual GroundingとReferring Expression**

ユーザーのタスクは、一般的な「物体検出（Object Detection）」とは異なります。特定のカテゴリ（例："person", "car"）を見つけるのではなく、「"a man sitting on a bench"（ベンチに座っている男性）」や「"the small scratch on the left"（左側にある小さな傷）」といった自由形式の自然言語（指示表現）によって記述された特定のインスタンスをローカライズする必要があります。

このタスクは、専門的には**Visual Grounding**（またはReferring Expression Comprehension: REC）と呼ばれます 1。アノテーションがピクセル単位のマスクになる場合、特に\*\*Referring Expression Segmentation (RES)\*\*と呼ばれます 4。

### **1.2. タスク1：Bounding Box（バウンディングボックス）の生成**

言語で指定された異常領域を、矩形のバウンディングボックスとして自動アノテーションするモデルを分析します。

#### **1.2.1. 基盤モデル分析：Grounding DINO**

* **概要:** Grounding DINOは、オープン・ボキャブラリ物体検出（Open-Set Object Detection）の分野で広く認識されている、強力な基盤モデルです 12。  
* **アーキテクチャ:** 物体検出モデルDINOと、テキストエンコーダ（例：BERT）を組み合わせ、画像特徴とテキスト特徴を融合させることで、テキストプロンプトに基づいた検出（Grounded Pre-training）を実現します 12。  
* **特徴:** transformers などのHugging Faceライブラリを通じて容易に利用可能であり（AutoModelForZeroShotObjectDetection）15、テキストプロンプトを入力として受け取り、検出されたオブジェクトのバウンディングボックスとスコアを出力します 14。  
* **概念コード（Hugging Face transformers）:**  
  Python  
  from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  
  from PIL import Image  
  import torch

  \# モデルとプロセッサのロード \[15\]  
  model\_id \= "IDEA-Research/grounding-dino-base"  
  processor \= AutoProcessor.from\_pretrained(model\_id)  
  model \= AutoModelForZeroShotObjectDetection.from\_pretrained(model\_id)

  image \= Image.open("path/to/generated\_abnormal\_image.png")  
  text\_prompt \= "a crack on the surface" \# ユーザーの異常指定プロンプト

  \# 入力の準備と推論 \[16\]  
  inputs \= processor(images=image, text=text\_prompt, return\_tensors="pt")  
  with torch.no\_grad():  
      outputs \= model(\*\*inputs)

  \# 結果の後処理 \[17, 75\]  
  \# target\_sizes はリサイズを元に戻すために必要  
  results \= processor.post\_process\_object\_detection(outputs, target\_sizes=\[image.size\[::-1\]\], threshold=0.4)

  \# 'results\["boxes"\]' にバウンディングボックスの座標 (xyxy形式) が含まれる  
  \# 'results\["labels"\]' に対応するプロンプト内のテキストインデックスが含まれる

#### **1.2.2. 基盤モデル分析：Microsoft Florence-2**

* **概要:** Florence-2は、Microsoft Azure AIによって開発された、次世代の統一型ビジョン基盤モデル（Vision Foundation Model）です 19。  
* **アーキテクチャ:** 単一のEncoder-Decoderモデルが、テキストプロンプトによるタスク指定（Task-Prompting）を通じて、キャプション、物体検出、Visual Grounding、セグメンテーションなど、多様なタスクを統一的に処理します 19。  
* **特徴:** ユーザーがAzure環境（gpt-image-1）を既に利用している点で、Florence-2は技術的にも環境的にも最も親和性が高い選択肢です。Florence-2は、Visual Grounding 19 やReferring Expression Comprehension 19 タスクにおいて、既存の専門モデル（Grounding DINOを含む）を凌駕するゼロショット性能を達成しています。  
* **タスクプロンプト:** Florence-2でBounding BoxによるGroundingを実行するには、\<OPEN\_VOCABULARY\_DETECTION\> や \<CAPTION\_TO\_PHRASE\_GROUNDING\> といった専用のタスクプロンプトを、ユーザーのテキスト入力と組み合わせて使用します 21。  
* **性能:** Florence-2は、Grounding DINOと比較して、Visual Groundingタスクにおける信頼性（faithfulness）が大幅に向上しているとの報告があり 23、より正確なローカライズが期待できます。

#### **1.2.3. 比較分析と推奨（Bounding Box）**

ユーザーがBounding Boxを生成するモデルを選択する際のトレードオフを以下に示します。

| モデル名 | アーキテクチャ | 主なタスク | ゼロショット性能 | 柔軟性（他タスク） | 実装（Hugging Face） |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Grounding DINO** 12 | 検出特化型 (DINO+BERT) | オープン・ボキャブラリ検出 (Box) | 高精度 \[13\] | 低（検出特化） | 容易 \[15\] |
| **Florence-2** 19 | 統一型VLM (Encoder-Decoder) | 全タスク (Box, Mask, Caption等) \[22\] | **SOTA** 19 | **非常に高い** | 容易 |

推奨:  
Bounding Box生成のタスクにおいて、Florence-2を第一に推奨します。SOTA（State-of-the-Art）の性能 19、単一モデルでセグメンテーションまで対応可能な柔軟性 21、そしてユーザーの既存Azure環境との高い親和性 19 を併せ持つため、最も戦略的な選択と言えます。

### **1.3. タスク2：セグメンテーションマスクの生成**

Bounding Boxよりも詳細な、ピクセル単位の異常領域（マスク）を生成するアプローチです。これは、異常検出データセットのアノテーションとして最も高品質な形式です。

#### **1.3.1. パイプライン・アプローチ：Grounded-SAM (DINO \+ SAM)**

* **概要:** テキストプロンプトからセグメンテーションマスクを生成する手法として、現在、オープンソースコミュニティで最も広く使われている強力なパイプラインです 24。  
* **2モデルの連動:** このアプローチは、2つの異なる基盤モデルの強みを組み合わせたものです。  
  1. **Segment Anything Model (SAM)** 27 は、非常に強力なセグメンテーションモデルですが、**テキストプロンプトを直接受け取れません** 28。SAMは、点（Point）やBounding Boxといった「空間的プロンプト」を必要とします 29。  
  2. **Grounding DINO**（1.2.1節）は、テキストプロンプト（例：「"a crack"」）を解釈し、その「空間的プロンプト」であるBounding Boxを高精度で出力できます 24。  
* **処理フロー:** Grounded-SAMパイプラインは、以下の処理を自動で実行します 26：  
  1. ユーザーがテキスト（例：「"a crack"」）をGrounding DINOに入力します。  
  2. Grounding DINOが "a crack" のBounding Boxを出力します。  
  3. そのBounding Boxを、SAMへのプロンプトとして入力します。  
  4. SAMがBounding Box内の "a crack" の詳細なセグメンテーションマスクを出力します。  
* **SAM 2の活用:** Meta AIによってリリースされたSegment Anything 2 (SAM 2\) 27 は、従来のSAM 1と比較してセグメンテーション精度が大幅に（最大6倍）向上しています 27。Grounded-SAMパイプラインのSAM部分をSAM 2に置き換える（DINO \+ SAM 2）ことで、アノテーション品質の大幅な向上が期待できます。

#### **1.3.2. 統合モデル・アプローチ：Florence-2 (Standalone)**

* **概要:** Florence-2は、Grounded-SAMのような2モデルのパイプラインを必要とせず、**単一のモデル**でテキストから直接セグメンテーションマスクを生成できます 21。  
* **タスクプロンプト:** このタスクは、Florence-2において\*\*\<REFERRING\_EXPRESSION\_SEGMENTATION\>\*\*という専用のプロンプトで呼び出されます 21。  
* **出力形式:** モデルは推論結果として、セグメンテーションマスク（ポリゴン座標のリスト）を含むテキストまたはJSON形式の文字列を返します 21。  
* **概念コード（Hugging Face transformers）:**  
  Python  
  from transformers import AutoProcessor, AutoModelForCausalLM  
  from PIL import Image  
  import torch

  \# モデルとプロセッサのロード (Florence-2)  
  model\_id \= "microsoft/Florence-2-large"  
  model \= AutoModelForCausalLM.from\_pretrained(model\_id, trust\_remote\_code=True)  
  processor \= AutoProcessor.from\_pretrained(model\_id, trust\_remote\_code=True)

  image \= Image.open("path/to/generated\_abnormal\_image.png")

  \# タスクプロンプトとテキストプロンプトを結合 \[32\]  
  task\_prompt \= "\<REFERRING\_EXPRESSION\_SEGMENTATION\>"  
  text\_input \= "a green car" \# ユーザーの異常指定プロンプト  
  prompt \= task\_prompt \+ text\_input

  inputs \= processor(text=prompt, images=image, return\_tensors="pt")

  \# generateメソッドで推論を実行  
  outputs \= model.generate(  
      \*\*inputs,   
      max\_new\_tokens=1024, \# マスクの座標は長くなる可能性がある  
      do\_sample=False  
  )

  \# 専用のpost\_process\_generationで結果をパース   
  results \= processor.post\_process\_generation(outputs, task=task\_prompt, image\_size=image.size)

  \# 'results' には  
  \# "Polygons" (座標リスト) や "labels" が含まれる 

* **利点:** この単一モデルアプローチは、Grounded-SAMパイプラインと比較して、システム構成がシンプルであり、推論遅延やコンポーネント管理の面で有利である可能性が高いです。

#### **1.3.3. ハイブリッド・アプローチ：Florence-2 \+ SAM 2 (SOTA)**

Grounded-SAM (DINO \+ SAM 1\) の成功 26、Florence-2のGrounding DINOに対する優位性 23、およびSAM 2のSAM 1に対する優位性 27 を組み合わせることで、現時点での理論上最強のパイプラインを構築できます。

* **処理フロー:**  
  1. Grounded-SAMの成功は、「SOTA検出器 ＋ SOTAセグメンタ」の組み合わせにあります 26。  
  2. 現行のSOTA検出器（Groundingタスク）は**Florence-2**です 19。  
  3. 現行のSOTAセグメンタ（プロンプトベース）は**SAM 2**です 27。  
  4. したがって、(1) ユーザーがテキストをFlorence-2の\*\*\<OPEN\_VOCABULARY\_DETECTION\>\*\*タスクに入力 → (2) Florence-2がBounding Boxを出力 → (3) そのBoxをSAM 2へのプロンプトとして入力 → (4) SAM 2が超高精細なマスクを出力、というパイプラインが、最も高品質な自動アノテーションを実現する可能性が最も高いです。  
* **実証:** この「Florence-2 \+ SAM 2」の組み合わせは、Grounded-SAM-2 というリポジトリ名でのデモ 33 や、Roboflow、ComfyUIなどのサードパーティ・プラットフォームによって、既に実装・検証が開始されています 27。

#### **1.3.4. 実践的実装：アノテーション・パイプラインの構築と保存**

テキストプロンプトからマスク（NumPy配列）を生成した後、それをアノテーションデータセットとして保存する必要があります。

* **Grounded-SAM (v1/v2) の実装:**  
  * grounded\_sam\_demo.py のような公開スクリプト（30）を利用するのが近道です。  
  * スクリプトは、input\_image（画像パス）、text\_prompt（異常指定）、output\_dir（保存先）などを引数として受け取ります 30。  
  * 内部の predict 関数が DINO (Box) と SAM (Mask) を順次呼び出し、最終的なマスク（NumPy配列）を返します 25。  
* **アノテーションの保存形式:**  
  * **COCO RLE形式:** インスタンスセグメンテーションの標準形式です 36。pycocotools ライブラリ（pip install pycocotools）37 を使用し、生成されたバイナリマスク（NumPy配列）をRun-Length Encoding (RLE) 形式にエンコードし、COCO形式のJSONファイルに保存します。  
  * **PNGマスク:** 最もシンプルな方法として、生成されたマスク（0と1のNumPy配列）を255倍して uint8 型に変換し、グレースケール（またはバイナリ）のPNG画像として保存します 39。アノテーションツール（例：Roboflow, CVAT, Label Studio）の多くは、この形式のマスク画像をインポートできます 39。

#### **1.3.5. 比較分析と推奨（Segmentation Mask）**

テキストから「Mask」を生成する主要な3つのアプローチを比較します。

| パイプライン名 | 構成モデル | プロセス | 精度（理論値） | 実装の複雑さ |
| :---- | :---- | :---- | :---- | :---- |
| **Grounded-SAM (v1)** \[24, 26\] | Grounding DINO \+ SAM 1 | 2モデル・2ステップ | 中〜高 | 中（実績豊富） |
| **Florence-2 (Standalone)** \[21, 31\] | Florence-2 (単一) | 1モデル・1ステップ (RESタスク) | 高 | 低 |
| **Florence-2 \+ SAM 2** \[27, 33\] | Florence-2 (Box) \+ SAM 2 (Mask) | 2モデル・2ステップ | **非常に高い** | 中〜高 |

**推奨:**

* **最高の精度（品質）を求める場合:** **Florence-2 \+ SAM 2** のハイブリッド・パイプライン 27 を推奨します。SOTA検出器とSOTAセグメンタの組み合わせにより、最も高品質なアノテーションが期待できます。  
* **実装のシンプルさを求める場合:** **Florence-2 (Standalone)** 31 を推奨します。単一モデルで完結するため、管理が容易です。  
* **豊富なドキュメントと実績を求める場合:** **Grounded-SAM (v1)** 30 が依然として良い出発点です。

---

## **第2部：戦略A（マネージド・サービス）：クラウドAPIによるアプローチ**

ユーザーが既にAzure環境を利用しているため、第1部のモデルをセルフホストする代わりに、クラウドAPIとして利用するアプローチを分析します。

### **2.1. Microsoft Azure AI Vision**

* **概要:** Azureの主要な画像認識サービスであり、ユーザーの既存環境です 40。  
* **Image Analysis 4.0 API:** 最新のSDK（azure-ai-vision-imageanalysis）41 は、REST API (2023-10-01) に基づいており、従来のOCR 42 や物体検出 41 に加え、高度なビジョンタスクをサポートしています 44。  
* **AzureとFlorence-2の関係:** 第1部でSOTAモデルとして特定された**Florence-2**は、Microsoft Azure AIのチームによって開発されたモデルです 19。Florence-2の学習データ（FLD-5B）の作成には、Azure AI ServicesのOCR API（76）などが活用されています。  
* **APIの選定:** この強固な関係性を踏まえると、Azure AI Vision APIの最新機能（Image Analysis 4.0）は、Florence-2モデル（またはその後継）によって強化されている可能性が極めて高いです 45。したがって、ユーザーはAzure環境を離れることなく、第1部で議論したSOTAのVisual Grounding性能を、マネージドAPIとして享受できる可能性が高いです。  
* **注意点:** 最新のSDK (v1.0.0-beta.1) では、Image Segmentation（背景除去）機能がGA REST API (2023-10-01) ではまだサポートされておらず、古いPreview REST API (2023-04-01-preview) を直接呼び出す必要がある、という記述があります 44。テキストベースのセグメンテーション（RES）を利用する場合、APIのバージョンとドキュメントを慎重に確認する必要があります。

### **2.2. Google Cloud Vertex AI**

* **概要:** Googleの統合AIプラットフォームです。  
* **Visual Grounding機能:** Google Researchは「FindIt」モデル 46 やGemini API 47 を通じて、テキストベースのローカライゼーション 46 やVisual Grounding 1 を強力にサポートしています。  
* **APIの挙動:** Gemini-2.0-flashなどのLLMが、テキストの推論トレースからVisual Groundingのキュー（"detect call"）を抽出し、Bounding Boxを返す機能が報告されています 47。

### **2.3. Amazon Web Services (AWS)**

* **概要:** AWSのAIサービス群です。  
* **Amazon Bedrock Data Automation:** 書類、画像、ビデオなどのマルチモーダルコンテンツからインサイトを抽出する新機能です 48。  
* **Visual Grounding機能:** このサービスは、出力の透明性を高めるために「Visual Grounding」と「Confidence Score」を明示的に提供します 48。  
* **Rekognition / SageMaker:** 従来のRekognitionに加えて、SageMaker 51 を通じてOFA（Visual Groundingをサポート 51）や、テキストベースの物体検出 52 を利用することも可能です。

### **2.4. クラウドサービス比較と戦略**

すべての主要クラウドプロバイダーが、Visual Grounding / テキストベース検出機能を、それぞれの最先端VLM（Florence, Gemini, Bedrock/Titan）を基盤として、急速にAPI製品化しています 19。

| クラウドサービス | 主要API | 基盤モデル（推定） | Visual Grounding機能 | 親和性 |
| :---- | :---- | :---- | :---- | :---- |
| **Microsoft Azure** | Azure AI Vision (Image Analysis 4.0) \[40, 41\] | **Florence-2** 19 | SOTA 19。セグメンテーションはPreview APIの可能性 44。 | **非常に高い** |
| **Google Cloud** | Vertex AI (Gemini API) \[1, 47\] | Gemini, FindIt 46 | APIによる推論とGroundingをサポート 47。 | 中 |
| **AWS** | Amazon Bedrock Data Automation \[48, 49\] | Titan等 | APIが「Visual Grounding」を明示的にサポート \[49, 50\]。 | 中 |

推奨:  
ユーザーは既にAzure環境にいます。そして、そのAzureが開発したFlorence-2 19 が、オープンソースで比較してもSOTAのVisual Grounding性能 19 を持っています。したがって、他クラウドに移行する積極的な理由はなく、Azure AI Visionの最新API 41、またはAzure ML Studio 19 でのFlorence-2モデルのセルフデプロイが、最も低コストかつ高性能なアプローチとなります。

---

## **第3部：戦略B（代替パラダイム）：教師なし異常検出（UAD）の活用**

このセクションでは、ユーザーの前提（異常画像を生成し、アノテーションする）を根本から見直す、より効率的なアプローチを提案します。

### **3.1. 戦略的ピボット：アノテーション不要の異常検出**

* UADワークフローの優位性:  
  ユーザーの現在のワークフロー（生成→注釈→学習）は、生成モデル（gpt-image-1）の品質とVLM（Florence-2）の注釈品質という2つの不確定要素に依存します。  
* しかし、ユーザーは「**正常画像をたくさんある**」という、異常検出タスクにおいて最も価値のある資産を保有しています。  
* 産業用 57 や医療用 7 の異常検出では、「正常」のパターンは均一だが、「異常」のパターンは無限に存在するという非対称性があります。  
* このシナリオでは、「正常とは何か」だけを学習する「One-Class Learning」または「教師なし異常検出（UAD）」が、SOTA性能を達成することが知られています 5。  
* UADモデル（例：PatchCore）は、学習（train）フェーズでは**正常画像のみ**を必要とします 58。  
* 推論（predict）フェーズでテスト画像（異常または正常）を入力すると、学習した「正常」の分布から逸脱するピクセル領域を自動的に特定し、anomaly\_map（ヒートマップ）またはpred\_mask（セグメンテーションマスク）として出力します 61。  
* **結論:**  
  1. ユーザーが保有する「正常画像」のみでUADモデルを学習させるだけで、ユーザーの最終目的である「**異常検出器**」（セグメンテーション出力付き）が**完成**します。  
  2. このアプローチは、**異常画像の生成とアノテーションのプロセス全体を不要**にします。  
  3. （皮肉なことに）この学習済みUADモデルは、戦略Aでgpt-image-1が生成した「異常画像」をpredictさせるだけで、その異常領域のpred\_maskを自動で出力するため、**戦略Aのための最強の自動アノテーションツールとしても機能**します。

### **3.2. 主要モデルとライブラリ：PatchCore, PaDiM, および anomalib**

#### **3.2.1. モデル解説：PatchCore**

* **概要:** UAD分野において、精度と速度のバランスが最も優れたSOTAモデルの一つです 6。  
* **仕組み:** ImageNetで事前学習済みのCNN（例：Wide-ResNet）を利用します。学習時（正常画像のみ）、中間層から抽出した特徴量（パッチ）を「メモリバンク（特徴量辞書）」として記憶します 67。  
* **推論:** テスト画像のパッチ特徴量を抽出し、メモリバンク内の最も近い正常特徴量との距離を計算します。この距離が「正常」の分布から大きく逸脱した場合、その領域を「異常」としてスコアリングします 67。  
* **PaDiM:** PatchCoreの関連モデルであり、同様のメモリバンクベースのアプローチを採用し、高い性能を示します 65。

#### **3.2.2. 実践的実装：anomalib ライブラリ**

* **概要:** Intelが主導する、教師なし異常検出のための包括的なPythonライブラリです 67。PatchCore 59、PaDiM 68、Generative Model 66 など、多数のSOTA UADモデルを統一的なインターフェースで提供します。  
* anomalib によるアノテーション・パイプライン構築（チュートリアル）:  
  以下に、anomalib を使ってUADモデルを学習させ、それを自動アノテーションツールとして使用する手順を示します。  
  * **ステップ1：インストールとインポート**  
    Python  
    \# pip install anomalib  
    from anomalib.models import Patchcore  
    from anomalib.engine import Engine  
    from anomalib.data import Folder

  * ステップ2：\[学習\] 正常画像のみでモデルを学習  
    anomalib の Folder データモジュールは、指定されたディレクトリ構造から正常画像を自動で読み込みます 59。  
    Python  
    \#  に基づくコード  
    datamodule \= Folder(  
        root="path/to/your/dataset", \# データセットのルート  
        normal\_dir="good",  \# 正常画像のみが含まれるディレクトリ名  
        abnormal\_dir=None,  \# 学習時には異常画像は不要  
        task="segmentation", \# ピクセルレベルの検出（セグメンテーション）を指定  
        image\_size=(256, 256)  
    )

    model \= Patchcore() \# SOTAモデルであるPatchCoreを選択  
    engine \= Engine()

    \# 正常画像のみを使って学習を実行  
    engine.train(datamodule=datamodule, model=model)  
    \# 学習済みモデル（.ckpt）とOpenVINO（.bin）が   
    \# 'results/Patchcore/run/weights/' 等に保存される

  * ステップ3：\[推論と注釈\] 生成された異常画像を推論し、マスクを取得  
    学習済みモデル（またはOpenVINOにエクスポートされたモデル）をロードし、predictを実行します。  
    Python  
    from anomalib.deploy import OpenVINOInferencer  
    from PIL import Image  
    import numpy as np

    \#  に基づくコード  
    inferencer \= OpenVINOInferencer(  
        path="results/Patchcore/run/weights/openvino/model.bin" \# ステップ2で生成されたモデル  
    )

    \# ユーザーがgpt-image-1で生成した異常画像  
    abnormal\_image\_path \= "path/to/generated\_abnormal\_image.png"  
    image \= np.array(Image.open(abnormal\_image\_path))

    \# 推論（これが自動アノテーションとなる）  
    predictions \= inferencer.predict(image=image)

    \# 結果の抽出   
    anomaly\_map \= predictions.anomaly\_map \# ヒートマップ (numpy array)  
    pred\_mask \= predictions.pred\_mask     \# バイナリマスク (numpy array, 0 or 1\)  
    pred\_score \= predictions.pred\_score   \# 画像レベルの異常スコア

  * ステップ4：\[保存\] 自動アノテーション（マスク）の保存  
    取得した pred\_mask（NumPy配列）を、cv2.imwrite などで画像ファイルとして保存します 73。  
    Python  
    import cv2  
    import os

    \#  に基づくコード  
    \# pred\_mask は (H, W) のnumpy array, 0 or 1  
    output\_mask \= (pred\_mask \* 255).astype(np.uint8)

    output\_dir \= "path/to/annotations"  
    os.makedirs(output\_dir, exist\_ok=True)  
    output\_path \= os.path.join(output\_dir, "generated\_abnormal\_mask.png")

    cv2.imwrite(output\_path, output\_mask)

#### **3.2.3. 比較分析（UADモデル）**

| モデル名 | 主要アプローチ | 学習データ | 推論出力 | 性能 (MVTec AD) |
| :---- | :---- | :---- | :---- | :---- |
| **PatchCore** \[6, 67\] | メモリバンク（特徴量辞書） | 正常のみ | Heatmap/Mask | SOTA \[66\] |
| **PaDiM** 65 | メモリバンク（ガウス分布） | 正常のみ | Heatmap/Mask | 高精度 \[65\] |
| **Generative Models** 66 | 再構成ベース (AutoEncoder, GAN) | 正常のみ | 再構成誤差→Heatmap | PatchCore等に劣る傾向 \[66\] |

---

## **結論と戦略的推奨**

### **4.1. アプローチの統合的評価**

* **戦略A（プロンプト・トゥ・アノテーション）:**  
  * **長所:** ユーザーの現在の要求に忠実です。異常が「セマンティック」（例：「ネジが欠けている」）または「既知」で、言語化しやすい場合に有効です。Grounded-SAM 26 や Florence-2 19 は、汎用的なアノテーションツールとして機能します。  
  * **短所:** 最終的なアノテーション品質が「①画像生成モデル（gpt-image-1）の品質」と「②VLM（Florence-2）の言語解釈・Grounding精度」という2つのステップに強く依存します。  
* **戦略B（教師なし異常検出）:**  
  * **長所:** ユーザーが保有する「正常画像」という最も信頼性の高い資産のみを使用します 59。異常画像の生成もアノテーションも**不要**です 5。特に「未知の異常」や「言語化しにくいテクスチャ異常」（例：微細な傷、ムラ）の検出において、SOTAの性能を発揮します 57。  
  * **短所:** 「正常」のばらつきが大きすぎる場合（例：製品ではなく自然風景）や、異常が「文脈的」（例：「あるべき場所にない」）である場合、性能が低下する可能性があります。

### **4.2. 専門家としての最終推奨（アクションプラン）**

ユーザーの状況（大量の正常画像を保有、gpt-image-1で異常画像を生成済み）を勘案し、以下の段階的なアクションプランを推奨します。

1. **\[最優先推奨\] 戦略B：anomalibによるUADモデルの構築（第3部）**  
   * **アクション:** まず、第3部で概説した\*\*戦略B（教師なし異常検出）\*\*を試行してください。anomalibライブラリ 71 をインストールし、保有する「正常画像」のみを使用してPatchCoreモデル 59 を学習させます。  
   * **理由:** これが、ユーザーの最終目的（異常検出器）への最短経路である可能性が最も高いです。このモデル自体が、異常画像の生成やアノテーションを一切不要にする、高性能な異常検出器（セグメンテーション出力付き 62）となります。  
2. **\[次善策\] 戦略Bを「アノテーションツール」として活用（ハイブリッド）**  
   * **アクション:** 戦略Bで学習したPatchCoreモデルを、ユーザーがgpt-image-1で生成した「異常画像」の**自動アノテーションツール**として使用します（第3.2.2項のステップ3・4）。  
   * **理由:** predictが返すpred\_mask 61 は、VLM（戦略A）が言語を解釈するよりも高精度に「正常からの逸脱」をピクセルレベルで捉える可能性があります。これは、VLMの「意味的」な理解と、UADの「統計的」な理解を組み合わせた、非常に強力なハイブリッド・アノテーション・パイプラインとなります。  
3. **\[代替案\] 戦略A：Florence-2 \+ SAM 2によるアノテーション（第1部）**  
   * **アクション:** 戦略Bが適さない場合（例：異常が言語的な文脈に強く依存し、UADモデルが検出できない場合）、第1部で概説した\*\*戦略A（プロンプト・トゥ・アノテーション）\*\*を実行します。  
   * **推奨技術スタック（セルフホスト）:** オープンソースの\*\*「Florence-2 \+ SAM 2」\*\* 27 のパイプラインを推奨します。これは、現行のSOTA検出器とSOTAセグメンタを組み合わせた、最高のセグメンテーション品質が期待できる構成です。  
   * **推奨技術スタック（クラウド）:** ユーザーのAzure環境を活用し、**Azure AI Vision API（Image Analysis 4.0）** 19 またはAzure ML Studioでの**Florence-2モデルのデプロイ** 56 を検討します（第2部）。

#### **引用文献**

1. Llama 3.2 API Service – Vertex AI – Google Cloud console, 11月 5, 2025にアクセス、 [https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.2-90b-vision-instruct-maas?inv=1\&invt=AbeNAg](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.2-90b-vision-instruct-maas?inv=1&invt=AbeNAg)  
2. Learning Visual Grounding from Generative Vision and Language Model \- CVF Open Access, 11月 5, 2025にアクセス、 [https://openaccess.thecvf.com/content/WACV2025/papers/Wang\_Learning\_Visual\_Grounding\_from\_Generative\_Vision\_and\_Language\_Model\_WACV\_2025\_paper.pdf](https://openaccess.thecvf.com/content/WACV2025/papers/Wang_Learning_Visual_Grounding_from_Generative_Vision_and_Language_Model_WACV_2025_paper.pdf)  
3. Learning Visual Grounding from Generative Vision and Language Model \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2407.14563v1](https://arxiv.org/html/2407.14563v1)  
4. Multimodal Referring Segmentation: A Survey \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2508.00265v1](https://arxiv.org/html/2508.00265v1)  
5. One D⁢i⁢n⁢o⁢m⁢a⁢l⁢y2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2510.17611v1](https://arxiv.org/html/2510.17611v1)  
6. MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2505.09265v1](https://arxiv.org/html/2505.09265v1)  
7. Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification \- CVF Open Access, 11月 5, 2025にアクセス、 [https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh\_Inter-Realization\_Channels\_Unsupervised\_Anomaly\_Detection\_Beyond\_One-Class\_Classification\_ICCV\_2023\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.pdf)  
8. Towards Understanding Visual Grounding in Vision-Language Models \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2509.10345v1](https://arxiv.org/html/2509.10345v1)  
9. Enabling and optimizing multi-modal sense-making for human-AI interaction tasks \- Institutional Knowledge (InK) @ SMU, 11月 5, 2025にアクセス、 [https://ink.library.smu.edu.sg/context/etd\_coll/article/1600/viewcontent/GPIS\_AY2019\_PhD\_DulangaWeerakoon.pdf](https://ink.library.smu.edu.sg/context/etd_coll/article/1600/viewcontent/GPIS_AY2019_PhD_DulangaWeerakoon.pdf)  
10. henghuiding/Awesome-Multimodal-Referring-Segmentation \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/henghuiding/Awesome-Multimodal-Referring-Segmentation](https://github.com/henghuiding/Awesome-Multimodal-Referring-Segmentation)  
11. CoT Referring: Improving Referring Expression Tasks with Grounded Reasoning \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2510.06243v1](https://arxiv.org/html/2510.06243v1)  
12. IDEA-Research/GroundingDINO: \[ECCV 2024\] Official implementation of the paper "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)  
13. Top 5 zero-shot object detection models in 2025 \- InteligenAI, 11月 5, 2025にアクセス、 [https://inteligenai.com/zero-shot-detection-enterprise/](https://inteligenai.com/zero-shot-detection-enterprise/)  
14. Zero-shot object detection with Grounding DINO | by Tauseef Ahmad | Medium, 11月 5, 2025にアクセス、 [https://medium.com/@tauseefahmad12/zero-shot-object-detection-with-grounding-dino-aefe99b5a67d](https://medium.com/@tauseefahmad12/zero-shot-object-detection-with-grounding-dino-aefe99b5a67d)  
15. Beyond the Demo: Dismantling Grounding Dino | by Neel Gahalot \- Medium, 11月 5, 2025にアクセス、 [https://medium.com/@ng2436/beyond-the-demo-dismantling-grounding-dino-63b145f42cb2](https://medium.com/@ng2436/beyond-the-demo-dismantling-grounding-dino-63b145f42cb2)  
16. Zero-shot Object Detection Using Grounding DINO Base \- Analytics Vidhya, 11月 5, 2025にアクセス、 [https://www.analyticsvidhya.com/blog/2024/10/grounding-dino-base/](https://www.analyticsvidhya.com/blog/2024/10/grounding-dino-base/)  
17. Grounding DINO \- Hugging Face, 11月 5, 2025にアクセス、 [https://huggingface.co/docs/transformers/v4.44.2/model\_doc/grounding-dino](https://huggingface.co/docs/transformers/v4.44.2/model_doc/grounding-dino)  
18. Fine-Tuning Grounding DINO: Open-Vocabulary Object Detection \- Learn OpenCV, 11月 5, 2025にアクセス、 [https://learnopencv.com/fine-tuning-grounding-dino/](https://learnopencv.com/fine-tuning-grounding-dino/)  
19. \[2311.06242\] Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks \- ar5iv, 11月 5, 2025にアクセス、 [https://ar5iv.labs.arxiv.org/html/2311.06242](https://ar5iv.labs.arxiv.org/html/2311.06242)  
20. Florence-2: Open Source Vision Foundation Model \- OpenVINO™ documentation, 11月 5, 2025にアクセス、 [https://docs.openvino.ai/2024/notebooks/florence2-with-output.html](https://docs.openvino.ai/2024/notebooks/florence2-with-output.html)  
21. microsoft / florence-2 \- NVIDIA API Documentation, 11月 5, 2025にアクセス、 [https://docs.api.nvidia.com/nim/reference/microsoft-florence-2](https://docs.api.nvidia.com/nim/reference/microsoft-florence-2)  
22. Florence-2 \- Roboflow Inference, 11月 5, 2025にアクセス、 [https://inference.roboflow.com/foundation/florence2/](https://inference.roboflow.com/foundation/florence2/)  
23. Interpreting Object-level Foundation Models via Visual Precision Search \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2411.16198v1](https://arxiv.org/html/2411.16198v1)  
24. Segmentation with Text Prompt: GroundingDINO+SAM2 \- Kaggle, 11月 5, 2025にアクセス、 [https://www.kaggle.com/code/patiencechewyeecheah/segmentation-with-text-prompt-groundingdino-sam2](https://www.kaggle.com/code/patiencechewyeecheah/segmentation-with-text-prompt-groundingdino-sam2)  
25. Zero-Shot Image Annotation with Grounding DINO and SAM \- A Notebook Tutorial, 11月 5, 2025にアクセス、 [https://blog.roboflow.com/enhance-image-annotation-with-grounding-dino-and-sam/](https://blog.roboflow.com/enhance-image-annotation-with-grounding-dino-and-sam/)  
26. Grounding-DINO \+ Segment Anything Model (SAM) vs Mask-RCNN: A comparison \- Encord, 11月 5, 2025にアクセス、 [https://encord.com/blog/grounding-dino-sam-vs-mask-rcnn-comparison/](https://encord.com/blog/grounding-dino-sam-vs-mask-rcnn-comparison/)  
27. What is Segment Anything 2 (SAM 2)? \- Roboflow Blog, 11月 5, 2025にアクセス、 [https://blog.roboflow.com/what-is-segment-anything-2/](https://blog.roboflow.com/what-is-segment-anything-2/)  
28. models/sam-2/ · ultralytics · Discussion \#14830 \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/orgs/ultralytics/discussions/14830](https://github.com/orgs/ultralytics/discussions/14830)  
29. How to Use the Segment Anything Model (SAM) \- Roboflow Blog, 11月 5, 2025にアクセス、 [https://blog.roboflow.com/how-to-use-segment-anything-model-sam/](https://blog.roboflow.com/how-to-use-segment-anything-model-sam/)  
30. SAM from Meta AI (Part 1): Segmentation with Prompts \- PyImageSearch, 11月 5, 2025にアクセス、 [https://pyimagesearch.com/2023/09/11/sam-from-meta-ai-part-1-segmentation-with-prompts/](https://pyimagesearch.com/2023/09/11/sam-from-meta-ai-part-1-segmentation-with-prompts/)  
31. florence-2-large model | Clarifai \- The World's AI, 11月 5, 2025にアクセス、 [https://clarifai.com/microsoft/florence/models/florence-2-large](https://clarifai.com/microsoft/florence/models/florence-2-large)  
32. Evaluating Visual-Language Models for Handwritten Text Recognition on Historical Swedish Manuscripts \- DiVA portal, 11月 5, 2025にアクセス、 [http://www.diva-portal.org/smash/get/diva2:1969262/FULLTEXT01.pdf](http://www.diva-portal.org/smash/get/diva2:1969262/FULLTEXT01.pdf)  
33. IDEA-Research/Grounded-SAM-2: Grounded SAM 2: Ground and Track Anything in Videos with Grounding DINO, Florence-2 and SAM 2 \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)  
34. ComfyUI-Workflow/awesome-comfyui: A collection of awesome custom nodes for ComfyUI \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/ComfyUI-Workflow/awesome-comfyui](https://github.com/ComfyUI-Workflow/awesome-comfyui)  
35. 11月 5, 2025にアクセス、 [https://huggingface.co/mart9992/nervn/commit/b793f0c366cc3f861e4e374d71beb2906bc9045e.diff](https://huggingface.co/mart9992/nervn/commit/b793f0c366cc3f861e4e374d71beb2906bc9045e.diff)  
36. Computer Vision Annotation Formats \- Roboflow, 11月 5, 2025にアクセス、 [https://roboflow.com/formats](https://roboflow.com/formats)  
37. ShoufaChen/Grounded-Segment-Anything-patch: Marrying Grounding DINO with Segment Anything & Stable Diffusion & BLIP \- Automatically Detect , Segment and Generate Anything with Image and Text Inputs \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/ShoufaChen/Grounded-Segment-Anything-patch](https://github.com/ShoufaChen/Grounded-Segment-Anything-patch)  
38. IDEA-Research/Grounded-Segment-Anything: Grounded SAM: Marrying Grounding DINO with Segment Anything & Stable Diffusion & Recognize Anything \- Automatically Detect , Segment and Generate Anything \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)  
39. How to Label Data with Grounded SAM 2 \- Roboflow Blog, 11月 5, 2025にアクセス、 [https://blog.roboflow.com/label-data-with-grounded-sam-2/](https://blog.roboflow.com/label-data-with-grounded-sam-2/)  
40. Azure AI Vision documentation \- Quickstarts, Tutorials, API Reference \- Microsoft Learn, 11月 5, 2025にアクセス、 [https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/)  
41. Azure Image Analysis client library for Python | Microsoft Learn, 11月 5, 2025にアクセス、 [https://learn.microsoft.com/en-us/python/api/overview/azure/ai-vision-imageanalysis-readme?view=azure-python](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-vision-imageanalysis-readme?view=azure-python)  
42. Getting Started with Microsoft Azure Computer Vision API in Python (Part 2: Handwriting Extraction) \- YouTube, 11月 5, 2025にアクセス、 [https://www.youtube.com/watch?v=7A38m5Dayk8](https://www.youtube.com/watch?v=7A38m5Dayk8)  
43. Quickstart: Optical character recognition (OCR) \- Azure AI services | Microsoft Learn, 11月 5, 2025にアクセス、 [https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/quickstarts-sdk/client-library](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/quickstarts-sdk/client-library)  
44. Image Analysis SDK Overview \- Azure AI services \- Microsoft Learn, 11月 5, 2025にアクセス、 [https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/sdk/overview-sdk](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/sdk/overview-sdk)  
45. Unified 3D Perception and Generative Control for Generalist Robots, 11月 5, 2025にアクセス、 [https://www.ri.cmu.edu/app/uploads/2025/08/ngkanats\_phd\_ri\_2025.pdf](https://www.ri.cmu.edu/app/uploads/2025/08/ngkanats_phd_ri_2025.pdf)  
46. Google Research, 2022 & beyond: Language, vision and generative models, 11月 5, 2025にアクセス、 [https://research.google/blog/google-research-2022-beyond-language-vision-and-generative-models/](https://research.google/blog/google-research-2022-beyond-language-vision-and-generative-models/)  
47. v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2505.18842v4](https://arxiv.org/html/2505.18842v4)  
48. Amazon Rekognition – Artificial Intelligence, 11月 5, 2025にアクセス、 [https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-rekognition/feed/](https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-rekognition/feed/)  
49. Unleashing the multimodal power of Amazon Bedrock Data Automation to transform unstructured data into actionable insights | Artificial Intelligence, 11月 5, 2025にアクセス、 [https://aws.amazon.com/blogs/machine-learning/unleashing-the-multimodal-power-of-amazon-bedrock-data-automation-to-transform-unstructured-data-into-actionable-insights/](https://aws.amazon.com/blogs/machine-learning/unleashing-the-multimodal-power-of-amazon-bedrock-data-automation-to-transform-unstructured-data-into-actionable-insights/)  
50. Amazon Transcribe – Artificial Intelligence, 11月 5, 2025にアクセス、 [https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-transcribe/feed/](https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/amazon-transcribe/feed/)  
51. Introducing an image-to-speech Generative AI application using Amazon SageMaker and Hugging Face | Artificial Intelligence, 11月 5, 2025にアクセス、 [https://aws.amazon.com/blogs/machine-learning/introducing-an-image-to-speech-generative-ai-application-using-amazon-sagemaker-and-hugging-face/](https://aws.amazon.com/blogs/machine-learning/introducing-an-image-to-speech-generative-ai-application-using-amazon-sagemaker-and-hugging-face/)  
52. Enhance video understanding with Amazon Bedrock Data Automation and open-set object detection | Artificial Intelligence, 11月 5, 2025にアクセス、 [https://aws.amazon.com/blogs/machine-learning/enhance-video-understanding-with-amazon-bedrock-data-automation-and-open-set-object-detection/](https://aws.amazon.com/blogs/machine-learning/enhance-video-understanding-with-amazon-bedrock-data-automation-and-open-set-object-detection/)  
53. Emergent Visual Grounding in Large Multimodal Models Without Grounding Supervision \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2410.08209v2](https://arxiv.org/html/2410.08209v2)  
54. Comprehensive review of recent developments in visual object detection based on deep learning \- ResearchGate, 11月 5, 2025にアクセス、 [https://www.researchgate.net/publication/392628462\_Comprehensive\_review\_of\_recent\_developments\_in\_visual\_object\_detection\_based\_on\_deep\_learning](https://www.researchgate.net/publication/392628462_Comprehensive_review_of_recent_developments_in_visual_object_detection_based_on_deep_learning)  
55. A Guide to Object Detection with Vision-Language Models | DigitalOcean, 11月 5, 2025にアクセス、 [https://www.digitalocean.com/community/conceptual-articles/hands-on-guide-to-object-detection-with-vision-language-models](https://www.digitalocean.com/community/conceptual-articles/hands-on-guide-to-object-detection-with-vision-language-models)  
56. Machine Learning Systems | PDF | Applied Mathematics | Computational Neuroscience, 11月 5, 2025にアクセス、 [https://www.scribd.com/document/880619817/Machine-Learning-Systems](https://www.scribd.com/document/880619817/Machine-Learning-Systems)  
57. M-3LAB/awesome-industrial-anomaly-detection \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/M-3LAB/awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)  
58. MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning \- NIPS papers, 11月 5, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf)  
59. Custom Data \- Anomalib documentation \- Read the Docs, 11月 5, 2025にアクセス、 [https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/how\_to/data/custom\_data.html](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/how_to/data/custom_data.html)  
60. Anomalib \- Release 2022 Intel OpenVINO, 11月 5, 2025にアクセス、 [https://anomalib.readthedocs.io/\_/downloads/en/v1/pdf/](https://anomalib.readthedocs.io/_/downloads/en/v1/pdf/)  
61. enrico310786/Image\_Anomaly\_Detection: Train and test image anomaly detection models with Anomalib. Examples on a custom dataset \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/enrico310786/Image\_Anomaly\_Detection](https://github.com/enrico310786/Image_Anomaly_Detection)  
62. Anomalib in 15 Minutes, 11月 5, 2025にアクセス、 [https://anomalib.readthedocs.io/en/latest/markdown/get\_started/anomalib.html](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html)  
63. Anomalib: a library for image anomaly detection and localization | by Enrico Randellini, 11月 5, 2025にアクセス、 [https://medium.com/@enrico.randellini/anomalib-a-library-for-image-anomaly-detection-and-localization-fb363639104f](https://medium.com/@enrico.randellini/anomalib-a-library-for-image-anomaly-detection-and-localization-fb363639104f)  
64. Anomaly Detection with FiftyOne and Anomalib, 11月 5, 2025にアクセス、 [https://docs.voxel51.com/tutorials/anomaly\_detection.html](https://docs.voxel51.com/tutorials/anomaly_detection.html)  
65. AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection | Request PDF \- ResearchGate, 11月 5, 2025にアクセス、 [https://www.researchgate.net/publication/391986047\_AnomalousPatchCore\_Exploring\_the\_Use\_of\_Anomalous\_Samples\_in\_Industrial\_Anomaly\_Detection](https://www.researchgate.net/publication/391986047_AnomalousPatchCore_Exploring_the_Use_of_Anomalous_Samples_in_Industrial_Anomaly_Detection)  
66. DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE CORSO DI LAUREA MAGISTRALE IN Control Systems Engineering “COMPARATIVE, 11月 5, 2025にアクセス、 [https://thesis.unipd.it/retrieve/8d7e427e-f828-4f0a-ac73-3d0ba690d6f6/Bugarin\_Nikola.pdf](https://thesis.unipd.it/retrieve/8d7e427e-f828-4f0a-ac73-3d0ba690d6f6/Bugarin_Nikola.pdf)  
67. PAEDID: Patch Autoencoder Based Deep Image Decomposition For Pixel- level Defective Region Segmentation \- SciSpace, 11月 5, 2025にアクセス、 [https://scispace.com/pdf/paedid-patch-autoencoder-based-deep-image-decomposition-for-2wke9im8.pdf](https://scispace.com/pdf/paedid-patch-autoencoder-based-deep-image-decomposition-for-2wke9im8.pdf)  
68. Anomaly Detection with FiftyOne and Anomalib, 11月 5, 2025にアクセス、 [https://docs.voxel51.com/tutorials/anomaly\_detection.html?highlight=anomalib](https://docs.voxel51.com/tutorials/anomaly_detection.html?highlight=anomalib)  
69. A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold \- arXiv, 11月 5, 2025にアクセス、 [https://arxiv.org/html/2211.12353v3](https://arxiv.org/html/2211.12353v3)  
70. Comparison of unsupervised image anomaly detection models for sheet metal glue lines \- DiVA portal, 11月 5, 2025にアクセス、 [https://www.diva-portal.org/smash/get/diva2:1958411/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1958411/FULLTEXT01.pdf)  
71. open-edge-platform/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference. \- GitHub, 11月 5, 2025にアクセス、 [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)  
72. A survey of deep learning for industrial visual anomaly detection \- ResearchGate, 11月 5, 2025にアクセス、 [https://www.researchgate.net/publication/392698522\_A\_survey\_of\_deep\_learning\_for\_industrial\_visual\_anomaly\_detection](https://www.researchgate.net/publication/392698522_A_survey_of_deep_learning_for_industrial_visual_anomaly_detection)  
73. Integrations, Plugins, and Model Evaluation — FiftyOne 1.8.1 documentation \- Voxel51, 11月 5, 2025にアクセス、 [https://docs.voxel51.com/getting\_started/manufacturing/05\_evaluation.html](https://docs.voxel51.com/getting_started/manufacturing/05_evaluation.html)  
74. See raw diff \- Hugging Face, 11月 5, 2025にアクセス、 [https://huggingface.co/spaces/blanchon/MVTec\_Padim\_Anomalib\_Test/commit/c8c12e9bd6d2898ea4e9f6a280a849dbf466054b.diff](https://huggingface.co/spaces/blanchon/MVTec_Padim_Anomalib_Test/commit/c8c12e9bd6d2898ea4e9f6a280a849dbf466054b.diff)  
75. Papers Explained 214: Florence-2. While existing large vision models… | by Ritvik Rastogi, 11月 5, 2025にアクセス、 [https://ritvik19.medium.com/papers-explained-214-florence-2-c4e17246d14b](https://ritvik19.medium.com/papers-explained-214-florence-2-c4e17246d14b)

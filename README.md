# Bridging Gaps in Hate Speech Detection: Meta-Collections and Benchmarks for Low-Resource Iberian Languages

![Language Support](https://img.shields.io/badge/Language-Multilingual-blue) 

A research-oriented repository on **hate speech detection** for Iberian languages 🌍, focusing on dataset construction, translation, experiments with transformers models, and linguistic analyses.

---

## 📂 Repository Structure

This repo is organized into four main sections:

### 1. 🏗️ `construction/`
How the **metacollection** was built:
- Aggregation of existing Spanish hate speech corpora  
- Cleaning, normalization, and deduplication  
- Metadata curation (content type, source)  
- Ensuring **consistency across datasets**  

📌 Goal: Provide a **unified benchmark resource** for Spanish hate speech research.  



### 2. 🌐 `translation/`
How the original **Spanish corpora** were translated into **Portuguese** and **Galician**:
- Machine Translation:
  - Using easy-translate from Spanish to European Portuguese (model `utter-project/EuroLLM-9B-Instruct`)

📌 Goal: Create **parallel datasets** to enable research in **low-resource settings**, where annotated data is scarce.  


### 3. 🔬 `experiments/`
Experiments with **NLP models** for hate speech detection:
- **BERT-based models** (multilingual)  
- **Large Language Models (LLMs)**: fine-tuning & prompting  
- **Zero-shot**, **few-shot**, and **cross-lingual transfer** setups  
- Benchmarking on unified datasets  

📌 Goal: Evaluate **robustness & generalization** across Iberian languages.  


### 4. 🧩 `analysis/`
Lexical and psycholinguistic analysis of the datasets:

#### 🔤 Lexical
- **NER**: Named Entity Recognition  
- **BERTopic**: Transformer-based topic modelling for semantic clustering  
- Word frequency distributions & lexical diversity  

#### 🧠 Psycholinguistic
- Pronoun and verb usage (e.g., personal pronouns, verb tenses)  
- **NRClex** emotion analysis across categories:  
  - *fear*, *anger*, *anticipation*, *trust*, *surprise*,  
  - *positive*, *negative*, *sadness*, *disgust*, *joy*  

📌 Goal: Understand **linguistic & psychological patterns** in hate speech, complementing model performance with deeper insights.  

---

## <img src="https://huggingface.co/favicon.ico" alt="Hugging Face" width="28" style="vertical-align: middle;"/> Datasets

Data can be accessed [here](https://huggingface.co/datasets/irlab-udc/MetaHateES)

---

## 📑 Citation

Coming soon!

---

## ⚠️ Disclaimer 

This repository includes content that may contain hate speech, offensive language, or other forms of inappropriate and objectionable material. The content present in the dataset or code is not created or endorsed by the authors or contributors of this project. It is collected from various sources and does not necessarily reflect the views or opinions of the project maintainers.  The purpose of using this repository is for research, analysis, or educational purposes only. The authors do not endorse or promote any harmful, discriminatory, or offensive behavior conveyed in the dataset.

Users are advised to exercise caution and sensitivity when interacting with or interpreting the repository. If you choose to use the datasets or models, it is recommended to handle the content responsibly and in compliance with ethical guidelines and applicable laws.  The project maintainers disclaim any responsibility for the content within the repository and cannot be held liable for how it is used or interpreted by others.

---

## 🙏 Acknowledgements 🙏

The authors thank the funding from the Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. The authors thank the financial support supplied by the grant PID2022-137061OB-C21 funded by MI-CIU/AEI/10.13039/501100011033 and by “ERDF/EU”. The authors also thank the funding supplied by the Consellería de Cultura, Educación, Formación Profesional e Universidades (accreditations ED431G 2023/01 and ED431C 2025/49) and the European Regional Development Fund, which acknowledges the CITIC, as a center accredited for excellence within the Galician University System and a member of the CIGUS Network, receives subsidies from the Department of Education, Science, Universities, and Vocational Training of the Xunta de Galicia. Additionally, it is co-financed by the EU through the FEDER Galicia 2021-27 operational program (Ref. ED431G 2023/01).

---

## 📜 License 

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

The Apache License 2.0 is an open-source license that allows you to use the software for any purpose, to distribute it, to modify it, and to distribute modified versions of the software under the terms of the license.

For more details, please refer to the Apache License 2.0.

---

## 📬 Contact 

For further questions, inquiries, or discussions related to this project, please feel free to reach out via email:

- **Email:** [paloma.piot@udc.es](mailto:paloma.piot@udc.es)

If you encounter any issues or have specific questions about the code, we recommend opening an [issue on GitHub](github.com/mlhate) for better visibility and collaboration.

# Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates

<!-- TOC -->
* [Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates](#retrieval-augmented-generation-for-argumentative-fallacies-in-us-political-debates)
  * [🚀 Overview](#-overview)
  * [🎯 Tasks](#-tasks)
  * [⛁ Knowledge Base](#-knowledge-base)
    * [Knowledge Base Format](#knowledge-base-format)
<!-- TOC -->

## 🚀 Overview

**Fallacies** are arguments that rely on faulty or misleading reasoning [(Goffredo et al., 2023)](https://aclanthology.org/2023.emnlp-main.684.pdf).

The goal of this project is to improve **fallacy detection and classification** in political discourse using **Retrieval-Augmented Generation (RAG)**. 
By grounding large language models in external knowledge sources, the system aims to provide more accurate classifications and clearer explanations of fallacious arguments in US political debates.

Experiments are conducted using the **[ElecDeb60to20](https://3ia-data.inria.fr/en/#)** dataset, which contains transcripts from U.S. political debates from 1960 to 2020.

## 🎯 Tasks

The project focuses on three main tasks.

- **Fallacy Detection**: detect whether a segment of speech contains a **fallacious argument**.
- **Fallacy Classification**: given a sentence already identified as fallacious, classify which type of fallacy it
  contains. Unlike detection, this task never sees non-fallacious sentences (`NoFallacy` sentences are dropped during
  dataset construction). The system currently supports the following fallacy categories:
    - _Ad Hominem_: an argument that attacks the opponent personally instead of addressing their position or reasoning.
    - _Appeal to Emotion_: the use of emotionally charged language intended to manipulate the audience’s feelings rather
      than provide logical support.
    - _Appeal to Authority_: an argument that relies on the endorsement of an authority figure or perceived consensus
      without sufficient evidence. It may also involve citing non-experts or majority opinion to support a claim.
    - _Slippery Slope_: an argument suggesting that a relatively small first step will inevitably lead to a chain of
      extreme or unlikely consequences.
    - _False Cause_: the incorrect assumption that a causal relationship exists between two events simply because they
      occur together.
    - _Slogan_: a short and striking phrase designed to provoke enthusiasm or agreement rather than present a logical
      argument.

## ⛁ Knowledge Base

To ground the language model, we constructed a **knowledge base of approximately 5GB of political and contextual
documents**. The corpus combines official government sources with curated background information.
The knowledge base is organized into seven main categories.

- **Bills and Laws**: this collection includes congressional bills and public and private laws approved by the U.S.
  Congress, collected from their official
  repository: [GovInfo](https://www.govinfo.gov/sitemaps), [U.S. Government Manual](https://www.govinfo.gov/app/collection/GOVMAN)
  and [Universal Declaration of Human Rights](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf)
- **Geography**: geographic and demographic information about U.S. states was collected from
  the [U.S. Census](https://data.census.gov/webpages)and Wikipedia.
- **Newspapers**: news articles providing political context for the debates, collected from
  the [Chronicling America](https://www.loc.gov/collections/chronicling-america/) collection at the Library of Congress
  and OCR-processed into text.
- **People**: biographical information about members of the U.S. Congress was collected from
  the [Biographical Directory of the United States Congress](https://bioguide.congress.gov/search). Additional entries
  from Wikipedia were added for public figures referenced during the debates.
- **History**: U.S. historical context, built from per-year Wikipedia "*{year} in the United States*" pages (from 1900
  onward) together with broad "*History of the United States*" era-overview pages and their linked articles. Each record
  is tagged with its year to support temporal filtering during retrieval.
- **Definitions**: conceptual definitions and explanations of argumentative fallacies were compiled from peer-reviewed
  research papers and reference
  sources ([Goffredo et al., 2023](https://aclanthology.org/2023.emnlp-main.684.pdf), [Goffredo et al., 2022](https://www.ijcai.org/proceedings/2022/0575.pdf)),
  a curated set of recent fallacy / argumentation papers retrieved from [OpenAlex](https://openalex.org/), and the
  Wikipedia fallacy pages.
- **Exemplars**: labeled examples of fallacious arguments drawn from public fallacy datasets
  ([LOGIC](https://huggingface.co/datasets/tasksource/logical-fallacy) and
  [MAFALDA](https://github.com/ChadiHelwe/MAFALDA)), mapped onto the project's fallacy taxonomy to provide concrete
  reference instances of each fallacy type.

### Knowledge Base Format

Each entry in the knowledge base is stored as a JSON document.

```json
{
  "id": "",
  "text": "",
  "xml": "[when available]",
  "source": "",
  "url": "",
  "year": "[when available]"
}
```

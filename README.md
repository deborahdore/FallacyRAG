# From Detection to Explanation: Retrieval-Augmented Generation for Argumentative Fallacies

## Overview

**Fallacies** are arguments that rely on faulty or misleading
reasoning [(Goffredo et al., 2023)](https://aclanthology.org/2023.emnlp-main.684.pdf).

The goal of this project is to improve **fallacy detection, classification, and explanation** in political discourse
using **Retrieval-Augmented Generation (RAG)**. By grounding large language models in external knowledge sources, the
system aims to provide more accurate classifications and clearer explanations of fallacious arguments in political
debates.

Experiments are conducted using the **[ElecDeb60to20](https://3ia-data.inria.fr/en/#)** dataset, which contains
transcripts from U.S. political debates.

---

# Tasks

The project focuses on three main tasks.

- **Fallacy Detection**: detect whether a segment of speech contains a **fallacious argument**.
- **Fallacy Classification**: classify the type of fallacy present in the argument. The system currently supports the
  following fallacy categories:
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
- **Fallacy Explanation**: generate a natural language explanation describing **why the argument was classified as a
  fallacy** and **why the reasoning is flawed**.

---

# Knowledge Base

To ground the language model, we constructed a **knowledge base of approximately 5GB of political and contextual
documents**. The corpus combines official government sources with curated background information.
The knowledge base is organized into four main categories.

- **Bills and Laws**: this collection includes congressional bills and public and private laws approved by the U.S.
  Congress, collected from their official
  repository: [GovInfo](https://www.govinfo.gov/sitemaps), [U.S. Government Manual](https://www.govinfo.gov/app/collection/GOVMAN)
  and [Universal Declaration of Human Rights](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf)
- **Geography**: geographic and demographic information about U.S. states was collected from
  the [U.S. Census](https://data.census.gov/webpages)and Wikipedia.
- **People**: biographical information about members of the U.S. Congress was collected from
  the [Biographical Directory of the United States Congress](https://bioguide.congress.gov/search). Additional entries
  from Wikipedia were added for public figures referenced during the debates.
- **Definitions**: conceptual definitions and explanations of argumentative fallacies were compiled from peer-reviewed
  research papers and reference
  sources ([Goffredo et al., 2023](https://aclanthology.org/2023.emnlp-main.684.pdf), [Goffredo et al., 2022](https://www.ijcai.org/proceedings/2022/0575.pdf)).
  Additional supporting definitions were collected from Wikipedia.

---

# Knowledge Base Format

Each entry in the knowledge base is stored as a JSON document.

### Standard Entry

```json
{
  "id": "",
  "text": "",
  "xml": "[when available]",
  "source": "",
  "source_url": ""
}
```

### Paragraph-Structured Entry

When the original document is already divided into paragraphs, the structure preserves this segmentation:

```json
{
  "id": "",
  "paragraph_n": "",
  "source_n": "",
  "source_url_n": "",
  "paragraph_n+1": "",
  "source_n+1": "",
  "source_url_n+1": ""
}
```

Each paragraph retains its corresponding source and reference URL to enable traceability during retrieval and
explanation generation.
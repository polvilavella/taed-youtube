---
annotations_creators:
- found
language:
- English
language_creators:
- found
license:
- cc0-1.0
multilinguality:
- monolingual
pretty_name: Youtube Statistics
size_categories:
- 10K<n<100K
source_datasets:
- original
tags: []
task_categories:
- text-classification
task_ids:
- sentiment-analysis
---

# Dataset Card for [Youtube Statistics]

## Table of Contents
- [Dataset Card for [Youtube Statistics]](#dataset-card-for-youtube-statistics)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://www.kaggle.com/datasets/advaypatil/youtube-statistics
- **Repository:** 
- **Paper:**
- **Leaderboard:**
- **Point of Contact:**

### Dataset Summary

This dataset contains two files, comments.csv with shape 4x18.4k and videos-stats.csv with shape 8x1880, for analyzing the relationship between the popularity of a certain video and the most relevant/liked comments of said video. We will only be using the comments.csv file to solve our task.

### Supported Tasks and Leaderboards

- 'sentiment-analysis': the goal of this task is to classify a given comment of a youtube video into positive, neutral or negative. There is no available leaderboard.

### Languages

English

## Dataset Structure

### Data Instances

```
{
    'Video ID': "wAZZ-UWGVHI",
    'Comment': "Let's not forget that Apple Pay in 2014 required a brand new iPhone in order to use it. A significa...",
    'Likes': 95.0,
    'Sentiment': 1.0
}
```

### Data Fields

- 'Video ID': String that identifies the video where the comment was written.
- 'Comment': String that contains the comment text.
- 'Likes': Integer representing the number of likes.
- 'Sentiment': Integer representing the sentiment of the comment (0: Negative, 1: Neutral, 2: Positive)

### Data Splits

There is no split between test and train sets.

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

Data may contain personally identifiable information, sensitive content, or toxic content that was publicly shared on the Internet.

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

[More Information Needed]

### Contributions

Thanks to [Advay Patil](https://www.kaggle.com/advaypatil) for adding this dataset.
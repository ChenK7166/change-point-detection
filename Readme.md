# Change Point Detection Model

This directory contains code for change point detection. To set up, create environment using `environment.yml`.

# How to run
Please call `annotate()` from *annotation.py*
```
def annotate(tweets, sampling_size=1000, feature_size=1000, min_range=86400, max_depth=3):
    """
    Args:
        messages: list of dictionaries [{'id':xxx,'contentText':xxx,...},...]
        sampling_size: sampling size of each date, default: 1000
        feature_size: tfidf feature size, default: 1000
        min_range: min range of detection, default: 86400
        max_depth: max depth of change point tree, default: 3

    Returns:
        A dict whose keys are message IDs and values are lists of annotations.
    
    Example return:
    [Annotation(
        id=message.id,
        type="change point",
        text=json.dumps({
            "datetime": 2022-06-22,
            "depth": 4
        }),
        confidence=0.7195,
        offsets=[],
        providerName="ta1-usc-isi"
        )
    Annotation(
        id=message.id,
        type="change point",
        text=json.dumps({
            "datetime": 2022-06-14,
            "depth": 3
        }),
        confidence=0.6772,
        offsets=[],
        providerName="ta1-usc-isi"
        )
    ...]
    """         
```

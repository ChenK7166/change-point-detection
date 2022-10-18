import numpy as np
import pandas as pd
import metachange
from model import ChangePointIndicator
import json
from incas_python.api.model.annotation import Annotation
from incas_python.api.model import Message, ListOfAnnotation


def annotate(message_file, sampling_size=1000, feature_size=1000, min_range=86400, max_depth=3):
    """
        Args:
            messages: list of dictionaries [{'id':xxx,'contentText':xxx,...},...]
            sampling_size: sampling size of each date, default: 1000
            feature_size: tfidf feature size, default: 1000
            min_range: min range of detection, default: 86400
            max_depth: max depth of change point tree, default: 3

        Returns:
            A dict whose keys are message IDs and values are lists of annotations.
    """
    # only use twitter dataset
    messages = [json.loads(line) for line in message_file]

    contentText_list = []
    timePublished_list = []
    for message in messages:
        if message["mediaType"] == "Twitter" or message["mediaTypeAttributes"]:
            if message["mediaTypeAttributes"]["twitterData"] != "null":
                contentText_list.append(message["contentText"])
                timePublished_list.append(int(message["timePublished"]))
            else:
                continue
        else:
            continue


    # for message in messages:
    #     contentText_list.append(message["contentText"])
    #     timePublished_list.append(int(message["timePublished"]))

    twitter_df = pd.DataFrame(list(zip(contentText_list, timePublished_list)),
                              columns=['contentText', 'timePublished'])

    changepoint_indicator = ChangePointIndicator(sampling_size, feature_size, min_range, max_depth)
    changepoint_indicator.run(twitter_df)

    annotations = {}
    for message in messages:
        id = message["id"]
        time = int(message["timePublished"])
        changepoint_results = changepoint_indicator.annotate(time)
        results = []
        for changepoint_result in changepoint_results:
            results.append(
                Annotation(
                    id=id,
                    type="change point",
                    text=json.dumps({
                        "datetime": changepoint_result[0].isoformat(),
                        "depth": changepoint_result[2]
                    }),
                    confidence=changepoint_result[1],
                    offsets=[],
                    providerName="ta1-usc-isi"
                )
            )
        annotations[id] = results

    return annotations

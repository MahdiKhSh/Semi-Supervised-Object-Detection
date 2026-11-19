# Reproducible qualitative failure analysis

Run RUN_VISUALIZE.cmd only after completing a native Windows core experiment.
This creates overlays on the genuine BDD100K validation images using the
trained model weights; it does not synthesize or alter the source data.

Save examples of true positives, missed objects and false positives across
night, rain and small/rare objects. Choosing examples exclusively to flatter
the model is a reproducibility failure: also include failure cases and random
examples. Manual inspection alone is not a COCO metric.

Compare the same image set across baseline and proposed checkpoints by
calling the script with --checkpoint to change models; keep threshold,
resolution and image order identical. Record checkpoint hashes and image
filenames in the paper. Never use synthetic smoke images as evidence of
road-detection performance.

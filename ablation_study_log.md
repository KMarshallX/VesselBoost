## Current pipeline
Register all images and labels as iterator

Put those iterators in a dict 

	For each epoch:

		For each iterator:

			crop a random patch -> augment

			For 1 ~ [batch_size]:
 
				Repeat line 10

			Concatenate the cropped patches [a batch]

			(training....)

## Test for Prediction

Train set: 4,5,6,7,8,9,11,12,14,18,19,20
Test set: 15,17

1. Train models with updated pipeline (a, b, c, d)
	set-up: epoch number -> 1k; number of subjects -> 12; batch multiplier -> 4; no augs
		number of patches totalling up to 48k (for a, b, c)
        
	set-up (d): epoch number -> 1k; number of subjects -> 12; batch multiplier -> 24; no augs
		number of patches totalling up to 288k

2. Predict on test set with the pre-trained models, respectively (a, b, c, d)

3. Evaluate the output segs with ground truths


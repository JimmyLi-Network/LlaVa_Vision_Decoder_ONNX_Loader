INSTRUCTIONS FOR THE STANDALONE SCRIPTS

Open a terminal in the directory and run the command:
python3 evaluate.py –g=gt/infographicVQA_val_v1.0.json –s=submissions/empty_results.json

parameters:
-g: Path of the Ground Truth file. The Ground Truth file is the one provided for the competition. You will be able to get it on the Downloads page of the Task in the Competition portal.
-s: Path of your method's results file.
 
Optional parameters:
-t: ANLS threshold. By default 0.5 is used. This can be used to check the tolerance to OCR errors. See Scene-Text VQA paper for more info.
-a: Boolean to get the ANLS break down by types. The ground truth file is required to have such information (currently on March 25th is not available to the public).
-o: Path to a directory where to copy the file 'results.json' that contains per-sample results.


Example: python3 evaluate.py –g=gt/infographicVQA_val_v1.0.json –s=submissions/empty_results.json

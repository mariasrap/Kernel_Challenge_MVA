# Kernel_Challenge_MVA

Repository with the code produced for the Kaggle Challenge in the context of the course Machine Learning with Kernel Methods. We mainly used a binary SVM classifier as a learning algorithm. We  tested the following kernels:

* n-th order walk kernel
* three versions of the random walk kernel



The code needed to reproduce the predictions used in the submission for the challenge (user María Sánchez del Río
) can be found in the file 'Submission_code.py' and can be executed using the command

```

python3 Submission_code.py

```

Note that you will need the precomputed kernel matrix from the file 'RWK_Labeled_4200.pkl'.You can also find the notebook version for a more clear view.

In this repository you can also find a notebook called 'Walk Kernels.ipynb' where I explain in detail the implementation of the different Kernels and the SVM.

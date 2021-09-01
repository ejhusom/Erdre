# Stage 1: profile

[Documentation - Home](https://github.com/SINTEF-9012/Erdre/blob/master/docs/index.md)

[Overview of pipeline](https://github.com/SINTEF-9012/Erdre/blob/master/docs/tutorials/03_pipeline.md)


The profile-stage creates a profile report of your data set. The report will
contain information on the number of variables, number of samples, statistical
properties of all the variables and correlation between variables, among other
things. The report may also generate warnings, for example if there are some
variables that contain a large amount of null-values, or if a variable stays
constant all the time. These warnings are used to clean the data in the next
stage of the pipeline.

The profile report is saved to the file `assets/profile/profile.html`, which
can be opened in a browser to give a comprehensive view of the properties of
your data set.


Next stage: [clean](https://github.com/SINTEF-9012/Erdre/blob/master/docs/tutorials/stages/02_clean.md)

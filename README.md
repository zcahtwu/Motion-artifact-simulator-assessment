# Uncovering errors in existing simulators of motion artefact for 3D MRI – assessment with analytical phantom (ISMRM 2026)

random_motion.py is a corrected version for TorchIO motion simulation, replace the random_motion.py file in TorchIO library with this version for correction. (Will try to make a pull request to TochIO library once all testing finishes.)

reguig_fix.txt provides an function for replacing their grid rotating function. (Note: This version is not fully tested, will complete the testing after the conference soon.)
Their function can be find in this file: https://github.com/romainVala/torchio/blob/master/src/torchio/transforms/augmentation/intensity/random_motion_from_time_course.py

k_space_based_simulation.py provides our own implementation of the k-space simulator, imoplementing the algorithm in the simulator by Reguig et al.

More detailed repo description comming soon.

(Updated 9th May 2026)

# Reproduction of "Data-driven Feature Tracking for Event Cameras"

Authors:

- Nicolas Chauvaux 925286
- Teun van der Ven 4962931
- Tjerk van der Weij 4858999
- Duncan van Woerkom 4878914



GitHub code: https://github.com/teungv/DL_group32/tree/main

## Introduction 
Existing feature trackers are currently primarily constrained by the hardware performance of standard frame-driven cameras. To be able to enhance this feature tracker performance, frame-driven cameras can be complemented by event-driven cameras to circumvent this limitation. Event-driven cameras are vision sensors that are able to capture information asynchronously, they only trigger when the change in brightness at a pixel exceeds a predetermined energy threshold. This allows them to generate sparse event streams with extremely high temporal resolution, efficiently capturing only the dynamic changes in a scene. This encapturement of dynamic changes makes them perfect for feature tracking. A visualization of this tracking is shown in the figure below. 
In this blog, we will reproduce the paper _Data-driven Feature Tracking for Event Cameras_ [1](https://arxiv.org/abs/2211.12826). The basic reproduction or training and inference without any change added to the code will be performed. Additionally we will test the performance of the model on inference with additional noise added to the dataset, testing its ability to adapt to and filter out irrelevant information, thereby showcasing its robustness in real-world scenarios where noise is inevitable. The model will also be tested on inference for different activation functions to evaluate how changes in the neural network's non-linearity affect its accuracy and responsiveness in tracking features under various conditions. 

![Figure 1 blog post summary](https://hackmd.io/_uploads/rkeUFi0JC.png)


### Setup
In the paper, feature tracking algorithms are used to track a given point in a reference frame in following timesteps. The model consists of a Feature Network and a Frame Attention module. The Feature Network processes each feature individually and uses a ConvLSTM layer with state `F` to process a correlation map `Cj` based on a template feature vector `R0` and the pixel-wise feature maps of the event patch. The Frame Attention module is used to combine the processed feature vectors for all tracks in an image using self-attention and a temporal state `S`, which is used to predict the relative feature displacement `delta Fj`. In figure .. the model is visualized from template and event patches of a greyscale image to the Feature Network, which outputs a single feature vector. This vector acts as the input for the Frame Attention Module that predicts the relative feature displacement.

![Screenshot 2024-04-06 at 12-49-53 2211.12826.pdf](https://hackmd.io/_uploads/rJZ-Ks0yR.png)

The network architecture details of the proposes model can be found in the figure below. Each convolutional layer is followed by a LeakyReLU and BatchNorm layers. Moreover, the linear layers are also followed by LeakyReLU layers. 


![Screenshot 2024-04-06 at 12-42-37 2211.12826.pdf](https://hackmd.io/_uploads/rJ75PiAJC.png)




### Training and Testing Datasets
For the training and evaluation of the model different datasets have been used. The model has been used with the Multiflow Dataset [[2]](https://arxiv.org/abs/2203.13674), which contains frames, synthetically generated events, and ground truth pixel flow. This dataset is free of noise and therefore methods are added to allow it to be applicable to real-life data.

As synthetic events differ drastically from real events, a novel pose supervision loss is proposed. This loss is based only on ground truth poses of a calibrated camera, which can be obtained by using for example the COLMAP algorithm. A triangulated point is computed based on the predicted track and the camera poses. They used this methodology to finetune the model obtained on the Multiflow dataset [2] using the Event Camera dataset (EC) [[3]](https://arxiv.org/abs/1610.08336) dataset. They also finetune it on the Event-aided Direct Sparse Odometry dataset (EDS) [[4]](https://arxiv.org/abs/2204.07640) dataset. However, in this blog post, we will only focus on the EC dataset and the corresponding results.

The testing and evaluation of the model is done on those same two datasets. The first being the EC dataset [3], that includes APS frames (24 Hz) and events. Additionally, the dataset provides ground truth camera poses at a rate of 200 Hz from an external motion. The second dataset used for testing is the newly published EDS dataset [4]. It is similar to the EC dataset, as it includes ground truth poses at a rate of 150 Hz from an external motion capture system. Both testing datasets were developed for the evaluation of camera pose estimation, resulting in mostly static scenes.

## Research questions

For the reproduction of the paper, we selected multiple criteria. First of all, we replicated the existing code implementation and tried to reproduce their results for both training and inference. Secondly, we evaluated different activation functions. And finally, the effect of adding noise to the event frames is evaluated.

The following research questions are tackled in this blog:
- "What is the effect of a new activation function on the benchmark results of the model?"
- "What is the effect of different types of noise on the benchmark results"

We expect that changing activation functions will decrease the performance of the model, as the model was trained on the unchanged initial activation function. For the second research question we expect it to perform poorly on low noise levels, due to the lack of training on data containing noise.

## Method
To test our research questions, first of all we start by reproducing the original paper [1]. For implementation, we used the existing code on GitHub [[5]](https://github.com/uzh-rpg/deep_ev_tracker/blob/master/models/correlation3_unscaled.py). We mainly followed the steps described in their README.md file. This proved to be fairly hard and we were hindered by several issues, as pointed out in the next Issues section.

Regarding training reproduction, as opposed to inference, the prepared pose training set is not provided on the GitHub page of the project. Therefore, we have downloaded the required poses from the initial dataset webpage [[6]](https://rpg.ifi.uzh.ch/davis_data.html). Which pose is needed is listed in table 3 of the paper. Then we have prepared the pose data required for fine-tuning by rectifying the data thanks to ``rectify_ec.py``, run colmap, and generate event representations with ``prepare_ec_pose_supervision.py``. For this, we followed the instruction "Preparing Pose Data" (https://github.com/uzh-rpg/deep_ev_tracker/tree/master#preparing-pose-data) section on GitHub. Finally, we started finetuning the model on the generated poses using the pre-training model obtained from the synthetic dataset following instructions in section "Training on Pose Data" (https://github.com/uzh-rpg/deep_ev_tracker/tree/master?tab=readme-ov-file#training-on-pose-data) from the GitHub page. To make sure hyper-parameters used are the same, we have contacted the authors to obtain confirmation.

For inference, we ran the model on the list of sequences defined in the EVAL_DATASETS variable in the ``evaluate_real.py`` file. This list contains five sequences of the EC dataset and five of the EDS dataset. As adviced, we ran the inference script on the sequences of the EC dataset. Once we had the predicted tracks for a sequence, we benchmarked their performance using the ``benchmark.py`` file. Here, the predicted tracks are compared against the re-projected, frame-based ground-truth tracks. The reproduction results can be found in the reproduction section. Moreover, even though the prepared pose data for testing sequence was provided by the authors on GitHub, we tried to regenerated them to verify full paper reproductibility by following instructions in section ''Preparing Input Data'' (https://github.com/uzh-rpg/deep_ev_tracker/tree/master?tab=readme-ov-file#preparing-input-data) of the GitHub page.

Then we made some minor adaptations to the code by changing the activation functions. In the code, we made modifications to two Python files: ``common.py`` and ``correlation3_unscaled.py``. We experimented with two different activation functions:
- Changing the LeakyReLU function to a ReLU function
- Increasing the slope of the LeakyReLU function

In figure below the difference between a standard ReLU activation function and a LeakyReLU function is visualized. A LeakyReLU function is used to address a limitation of a standard ReLU, which is the problem of dying neurons -neurons that output zero regardless of the input. LeakyReLU allows a small positive gradient to the negative part of the function, which can help during the backpropagation process and lead to more effective learning.

![new_act](https://hackmd.io/_uploads/rJNApnRyC.png) [https://paperswithcode.com/method/rrelu]

The ``common.py`` file defines layers following the convolutional layer. When changing the LeakyReLU function to a ReLU, in the common.py file, the ConvBlock class was adapted to implement a ReLU function. Each convolutional layer is now followed by a ReLU layer instead of a LeakyReLU layer. Also, in the correlation3_unscaled.py, the Attention Mask Transformer in the JointEncoder class defines was adjusted to contain two ReLU functions. The sequential container fusion_layer0 chains together two sets of linear transformation and ReLU activation layers. After the linear transformation, a non-linearity (ReLU function) is introduced. Moreover, the further transformation happening in the sequential container of fusion_layer0 is adapted to contain a nn.ReLU() function instead of a nn.LeakyReLU(0.1).

When increasing the slope of the LeakyReLU, we evaluated the effect of altering the proposed slope `0.10` to the larger value of `0.15` and thus steeper slope of the function for all inputs less than zero. The same files have been modified, only now changing the nn.LeakyReLU(0.1) to nn.LeakyReLU(0.15).


### Issues
During the reproduction of the project, numerous problems arose. Initially, it was expected that the code would easily be up and running. However, there were multiple factors that drastically slowed down this reproduction process. Below an overview of the problems with a brief description is listed.
- ReadMe file is not properly documented. This led to a lot of time spent on understanding the high and low level code. It was unclear why many of the choices and implementations were made and how they worked. This lack of proper documentation (also in the code sometimes) also increased the difficulty of other problems we faced in the reproduction model as it was not often clear how ever the program worked.
- Setting up the working environment with packages proved to be a minor issue. Due to the fact that the installation of certain packages with the`requirements.txt` file were not compatible anymore it was not possible to build the wheel for the installation. Partial separate installation solved this.
- It was not always clear how to manage certain paths in the code setup. This led to some minor solvable issues.
- The code required many adaptations to get up and running. Certain functions were deprecated and this required some code changes and other packages were not properly being found.
- For training part, many issues arised when regenerating the corrected pose training sequence from the initial EC dataset. The documentation explaining the steps to follow are very brief and doesn't explain much what is exactly happening. Following they instructions in section ''Preparing Pose Data'' (https://github.com/uzh-rpg/deep_ev_tracker/tree/master#preparing-pose-data):
    - The execution of ``rectify_ec.py`` ran without errors.
    - The execution of ``colmap.py`` failed with the following error: 
        ```console
        Traceback (most recent call last):
          File "<top_project_folder>\data_preparation\colmap.py", line 146, in <module>
            generate(sequence_dir=sequence_dir, dataset_type=dataset_type)
          File "<top_project_folder>\data_preparation\colmap.py", line 121, in generate
            image_pose = pose_interpolator.interpolate_colmap(image_ts)
          File "<top_project_folder>\utils\track_utils.py", line 306, in interpolate_colmap
            T_W_j[0, 3] = self.x_interp(t)
          File "<Package_dir>\site-packages\scipy\interpolate\_polyint.py", line 78, in __call__
            y = self._evaluate(x)

          File "<Package_dir>\site-packages\scipy\interpolate\_interpolate.py", line 695, in _evaluate
            below_bounds, above_bounds = self._check_bounds(x_new)
          File "<Package_dir>\site-packages\scipy\interpolate\_interpolate.py", line 724, in _check_bounds
            raise ValueError("A value in x_new is below the interpolation "
        ValueError: A value in x_new is below the interpolation range.
        ```
        We have contacted the authors regarding this error, but did not obtained it at the time they built the project.
    - As this error was not obtained for "poster_rotation", we have continued the instructions by runnong COLMAP. Step 1-4 ran without errors. Regarding step 5 (i.e. launch the colmap gui, import the model files, and re-run Bundle Adjustment ensuring that only extrinsics are refined.), some parameters need to be passed to COLMAP. As those parameters were not provided in the GitHub page, we used the default ones (see figure below):
    ![colmap_parameters](https://hackmd.io/_uploads/rkdRZ7FgA.png)
    However, COLMAP did not converge:
        ```console
        QBasicTimer::start: QBasicTimer can only be used with threads started with QThread
        QBasicTimer::start: QBasicTimer can only be used with threads started with QThread
        QObject::connect: Cannot queue arguments of type 'QItemSelection'
        (Make sure 'QItemSelection' is registered using qRegisterMetaType().)
        I20240327 12:25:55.640645 13352 misc.cc:198] 
        ==============================================================================
        Global bundle adjustment
        ==============================================================================
        I20240327 12:33:19.532618 13352 misc.cc:205] 
        Bundle adjustment report
        ------------------------
        I20240327 12:33:19.533618 13352 bundle_adjustment.cc:942] 
            Residuals : 1054046
           Parameters : 75752
           Iterations : 101
                 Time : 442.683 [s]
        Initial cost : 1.08664 [px]
           Final cost : 0.366978 [px]
          Termination : No convergence
        ```
        We have contacted the authors to obtain the parameters they used in COLMAP but they were not able to provide them to us.
    - The execution of ``prepare_ec_pose_supervision.py`` ran without errors.
    
    Despites all those errors, we have tried to start training with incomplete corrected train sequence. However, we again obtain a interpolation error:
    ```console
    Traceback (most recent call last):
      File "<top_project_folder>\train.py", line 72, in train
        trainer.fit(model, datamodule=data_module)
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 696, in fit
        self._call_and_handle_interrupt(
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 650, in _call_and_handle_interrupt
        return trainer_fn(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 735, in _fit_impl
        results = self._run(model, ckpt_path=self.ckpt_path)
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 1166, in _run
        results = self._run_stage()
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 1252, in _run_stage
        return self._run_train()
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 1283, in _run_train
        self.fit_loop.run()
      File "<Package_dir>\site-packages\pytorch_lightning\loops\loop.py", line 200, in run
        self.advance(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\fit_loop.py", line 271, in advance
        self._outputs = self.epoch_loop.run(self._data_fetcher)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\loop.py", line 200, in run
        self.advance(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\epoch\training_epoch_loop.py", line 203, in advance
        batch_output = self.batch_loop.run(kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\loop.py", line 200, in run
        self.advance(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\batch\training_batch_loop.py", line 87, in advance
        outputs = self.optimizer_loop.run(optimizers, kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\loop.py", line 200, in run
        self.advance(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 201, in advance
        result = self._run_optimization(kwargs, self._optimizers[self.optim_progress.optimizer_position])
      File "<Package_dir>\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 248, in _run_optimization
        self._optimizer_step(optimizer, opt_idx, kwargs.get("batch_idx", 0), closure)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 358, in _optimizer_step
        self.trainer._call_lightning_module_hook(
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 1550, in _call_lightning_module_hook
        output = fn(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\core\module.py", line 1674, in optimizer_step
        optimizer.step(closure=optimizer_closure)
      File "<Package_dir>\site-packages\pytorch_lightning\core\optimizer.py", line 168, in step
        step_output = self._strategy.optimizer_step(self._optimizer, self._optimizer_idx, closure, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\strategies\strategy.py", line 216, in optimizer_step
        return self.precision_plugin.optimizer_step(model, optimizer, opt_idx, closure, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\plugins\precision\precision_plugin.py", line 153, in optimizer_step
        return optimizer.step(closure=closure, **kwargs)
      File "<Package_dir>\site-packages\torch\optim\optimizer.py", line 140, in wrapper
        out = func(*args, **kwargs)
      File "<Package_dir>\site-packages\torch\optim\optimizer.py", line 23, in _use_grad
        ret = func(self, *args, **kwargs)
      File "<Package_dir>\site-packages\torch\optim\adam.py", line 183, in step
        loss = closure()
      File "<Package_dir>\site-packages\pytorch_lightning\plugins\precision\precision_plugin.py", line 138, in _wrap_closure
        closure_result = closure()
      File "<Package_dir>\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 146, in __call__
        self._result = self.closure(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 132, in closure
        step_output = self._step_fn()
      File "<Package_dir>\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 407, in _training_step
        training_step_output = self.trainer._call_strategy_hook("training_step", *kwargs.values())
      File "<Package_dir>\site-packages\pytorch_lightning\trainer\trainer.py", line 1704, in _call_strategy_hook
        output = fn(*args, **kwargs)
      File "<Package_dir>\site-packages\pytorch_lightning\strategies\strategy.py", line 358, in training_step
        return self.model.training_step(*args, **kwargs)
      File "<top_project_folder>\models\template.py", line 170, in training_step
        x_j, y_j = bl.get_next()
      File "<top_project_folder>\utils\dataset.py", line 1474, in get_next
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
      File "<top_project_folder>\utils\track_utils.py", line 294, in interpolate
        T_W_j[0, 3] = self.x_interp(t)
      File "<Package_dir>\site-packages\scipy\interpolate\_polyint.py", line 78, in __call__
        y = self._evaluate(x)
      File "<Package_dir>\site-packages\scipy\interpolate\_interpolate.py", line 695, in _evaluate
        below_bounds, above_bounds = self._check_bounds(x_new)
      File "<Package_dir>\site-packages\scipy\interpolate\_interpolate.py", line 724, in _check_bounds
        raise ValueError("A value in x_new is below the interpolation "
    ValueError: A value in x_new is below the interpolation range.
    ```
    
    To solve this issue, we have contacted the authors to obtain the data generated by ```rectify.py``` and COLMAP. We then tried to execute ```prepare_ec_pose_supervision.py``` on it and then run training. However, the same error was obtained then above indicating that something went wrong during execution of ```prepare_ec_pose_supervision.py```. We finally succeed to run training without error by requesting the full training sequence to the authors (the one being used to obtained results in the paper), therefore not needing to execute steps in section "Preparing Pose Data" (https://github.com/uzh-rpg/deep_ev_tracker/tree/master?tab=readme-ov-file#preparing-pose-data) of the GitHub page.



As a conclusion, when trying to reproduce the code, it will need extensive troubleshooting and some persistence, as multiple errors and obstacles are likely to emerge throughout the process.





## Results

### Reproduction corrected dataset

Regarding the test sequence, even though we have the already corrected sequence (marked as 'provided dataset' in the following tables), we wanted to check if we could regenerate it from the initial EC dataset (marked as 'generated dataset' in the following tables). We have download the original sequence from (https://rpg.ifi.uzh.ch/davis_data.html) and followed the instructions on section "Prepapring Input Data" (https://github.com/uzh-rpg/deep_ev_tracker/tree/master?tab=readme-ov-file#preparing-input-data) on the GitHub page. 
When comparing the provided and generated events, the two are different. In the next section, we have run inference and evaluation on the provided dataset and generated dataset and obtained a drop in Feature age (FA).

Regarding the training set, we were not able to regenrate it as per the errors obatined and explained in the section "Issues".

### Reproduction inference results

The table below shows the performance of the proposed tracker on the EC dataset (both provided and generated). The tracking metrics used are Feature Age (FA) and Expected Feature Age (Expected FA). Feature Age is the time until feature exceeds a certain distance to the ground-truth. To account for feature tracks immediately lost in the beginning, the ratio of stable tracks and ground-truth tracks is multiplied by the Feature Age, which gives the Expected Feature Age. In table below, the performance of the proposed tracker on the EC dataset is expressed in terms of Expected Feature Age.
 
| Sequence           | Paper (Expected FA)[1] | Ours - provided dataset (Expected FA)    | Ours - generated dataset (Expected FA)
|:------------------ | -------- | -------- | -------- |
| Shapes Translation |      0.856    |     0.854     | 0.729
| Shapes Rotation    |          0.793|          0.793| 0.685
| Shapes 6DOF        |          0.882|          0.879| 0.327
| Boxes Translation  |          0.869|          0.868| 0.476
| Boxes Rotation     |          0.691|          0.690| 0.302
| EC Avg             | 0.818 | 0.817|

Furthermore, in the table below, the performance of the proposed tracker on the EC dataset is expressed in terms of Feature Age.

 Sequence           | Paper (FA)[1] | Ours - provided dataset (FA)    | Ours - generated dataset (FA)
|:------------------ | -------- | -------- | -------- |
| Shapes Translation |      0.861    |     0.859     | 0.731
| Shapes Rotation    |          0.797|          0.797| 0.688
| Shapes 6DOF        |          0.899|          0.895| 0.361
| Boxes Translation  |          0.872|          0.871| 0.487
| Boxes Rotation     |          0.695|          0.693| 0.31
| EC Avg             | 0.825 | 0.823| 0.515

The tracking predictions can also be qualitatively shown. Here, the tracks for boxes_rotation_198_278 and for shapes_translation_8_88, respectively, can be seen.

![Screenshot 2024-04-06 at 15-54-31 boxes_rotation_198_278_tracks_pred.gif (GIF Image 640 × 480 pixels)](https://hackmd.io/_uploads/HynHVARk0.png)

![Screenshot 2024-04-06 at 15-49-15 shapes_translation_8_88_tracks_pred.gif (GIF Image 640 × 480 pixels)](https://hackmd.io/_uploads/BkTlE0RJA.png)

For shape rotation, and for our regenerated dataset, we can see that some points are inccorectly tracked, explaining the difference in FA compared to the provided dataset.

![tracking_fail](https://hackmd.io/_uploads/r1ShRmtg0.png)

### Reproduction training results

Using the training sequence that has been provided by the authors for the EC dataset (changing ```data: pose_ec``` and ```training: pose_finetuning_train_ec``` in ```train_defaults.yaml```) and starting from the pre-trained model on the synthetic dataset (by setting ```checkpoint_path``` in ```pose_finetuning_train_ec.yaml``` to ```weights_mf.ckpt```), we have performed training during 700 epochs (same as in the paper). Looking at the training and validation error, we can see that the model failed to learned correctly. 

![loss_train_epoch](https://hackmd.io/_uploads/BJkTtNteR.png)
![loss_val](https://hackmd.io/_uploads/HykpKVteR.png)

We evaluate our finetrained model on the test set and obtained very low FA. Looking at the figure for shapes_translation_8_88, the tracked points are all incorrect.

|        | Paper (Expected FA) | Our Finetrained Model (Expected FA) |
| ------ | ------------------- | --------- |
| EC Avg |                 0.818    |       0.011    |

|        | Paper (FA) | Our Finetrained Model (FA) |
| ------ | ------------------- | --------- |
| EC Avg |               0.825      |     0.0726      |

![tracking_fail_train](https://hackmd.io/_uploads/B1VWiVKxR.png)


### New activation function

Now we evaluate the benchmarked results of the new activation function. To be noted here that we always use the provided dataset and not out regenerated one. First of all, in the table below, the results of the new ReLU function are compared to the original results. As expected, the performance of the model significantly decreased when applying the new activation function.



|        | Paper (Expected FA)[1] | Ours - provided dataset (Expected FA) | Paper (FA)[1] | Ours - provided dataset (FA) |
| ------ | ------------------ | ------------------ | --------- | --------- |
| EC Avg |     0.818               |        0.725            |   0.825        |   0.731        |

Secondly, in the table below, the results of the steeper LeakyReLU function are compared to the original results. Again, the performance of the model significantly decreased when applying the new activation function. However, the results seem to be slightly better than with the ReLU function.

|        | Paper (Expected FA)[1] | Ours - provided dataset (Expected FA) | Paper (FA)[1] | Ours - provided dataset (FA) |
| ------ | ------------------ | ------------------ | --------- | --------- |
| EC Avg |     0.818               |        0.732            |   0.825        |   0.737        |


### Noise

To investigate the robustness of the model we decided to add noise to the data before running inference. This is done in prepare_ec_subseq.py.This file creates time surfaces which we altered to contain one of two types of noise. The two types of noise will be referred to as: "dropped" and "flipped" and will be explained below.

Flipped noise:
For this type of noise, a percentage of all pixels in the current batch is considered and if an element is not equal to 0 it is now set to 0 to remove the event. If the element is equal to 0 an event is created randomly somewhere else on the time surface. 

Dropped noise:
This type of noise, as the name suggests, drops a percentage of the events per batch. Dropping an event means that the polarity of the pixel where the selected event occurs is set to 0.
First, a list is filled with tuples containing row, column pairs of all events in that batch. The amount of events that have to be dropped is determined by multiplying the noise level with the amount of events that occur in the current batch. 
After this, the amount of events to drop are a random selection from the event indices. Then, the value of each of the selected row, column pairs in the time surface is set to 0.




Our added code:

```
# Add noise
noise_level = 0.1 # 0.1 = 10% of pixels flipped
noise_mode = "dropped" # choose between "flipped" and "dropped"

if noise_mode == "flipped":
                        for bin_val in range(2):
                            # Generate random indices to flip
                            all_indices = np.arange(IMG_H * IMG_W)
                            flip_indices = np.random.choice(all_indices, num_pixels_to_flip, replace=False)

                            # Convert the flattened index to row and column indices
                            row_index = flip_indices // IMG_W # in the first 240 pixels -> 1 -> row 1
                            col_index = flip_indices % IMG_W # in the 490 / 240 = 10 is the 10th column

                            # We are dealing with time surface. Therefore, if the index value is non zero, we sent it to 0, otherwise, we set it to a random correct value
                            for row, col in zip(row_index, col_index):
                                element = time_surface[row, col, i_bin + bin_val] # get the value of the event
                                if element != 0: # if the event is not zero, we set it to zero
                                    time_surface[row, col, i_bin + bin_val] = 0.0
                                else: # if the event is zero, we set it to a random time, so a random event
                                    noise_event_time = round(np.random.uniform(t0_bin, t1_bin), 6)
                                    time_surface[row, col, i_bin + bin_val] = noise_event_time - t0

                    elif noise_mode == "dropped":
                        for bin_val in range(2):

                            event_indices = np.array([
                            (row, col) for row in range(IMG_H) for col in range(IMG_W)
                            if time_surface[row, col, i_bin + bin_val] != 0
                                                    ])
                            number_of_events_to_drop = int(noise_level * len(event_indices))
                            drop_indices = np.random.choice(len(event_indices), number_of_events_to_drop, replace=False)

                            for index in drop_indices:
                                row, col = event_indices[index]
                                time_surface[row, col, i_bin + bin_val] = 0
```

All of the code was altered in such a way that only the variables "noise_mode" and "noise_level" had to be altered when running the robustness tests. The exact steps used for this test are shown below:

####    Noise steps

##### START
------------------------------------------------------------------------------
Data preparation:
1. Create a clean_data dir in ec_subseq with the original 5 video folders
2. Within prepare_ec_subseq.py:
2.1 sequence_root =  /your/path/to/clean_data
2.2 list_to_generate = [('boxes_rotation', 198, 278), ('boxes_translation', 330, 410), ('shapes_6dof', 485, 565), ('shapes_rotation', 165, 245), ('shapes_translation', 8, 88)]
2.3 Change noise level and mode
3. Run prepare_ec_subseq.py
4. create a noise_{noise_level} dir within ec_subseq
5. extract subseq directories from clean_data and copy to noise_{noise_level} dir

------------------------------------------------------------------------------
Inference:
1. create folder in gt_tracks/benchmark_data/noise_mode/noise_level copying gt and empyting network_pred to be filled later
2. evaluate_real.py -> change noise_level, noise_mode
3. eval_real_defaults.yaml -> change noise_level, noise_mode
5. Change noise level, noise_mode in dataset.py!
6. run evaluate_real.py

------------------------------------------------------------------------------
Evaluation

1. copy files from correlation to corresponding folder in gt_tracks/benchmark_data
2. create a folder in benchmark_results/noise_mode with the current noise level as name
3. benchmark.py -> change noise level
4. run benchmark.py

### Noise results

![noise_results_dropped](https://hackmd.io/_uploads/SJo7nIYe0.png)
![noise_results_flipped](https://hackmd.io/_uploads/Sksmh8FlR.png)

As we can see the model is not robust. At 1% noise, for both flipped and dropped noise, the model performance drops to about 25% percent of the original performance. This trend continues and plateaus to about 12,5% of the original performance at 10% noise for both cases. Flipped noise performs slightly worse when compared to dropped noise. We expect this to be the case due to the newly created events that appear with flipped noise having a very short lifespan pulling the expected age down even further. 
 
## Discussion
The project has been produced succesfully. There have been a few bumps in the road and this has led to more effort being put into the reproduction of the paper than actually adding and testing new modifications or exploring beyond the initial scope outlined in the original study.

The comparison made between the different activation functions; LeakyRelu, Relu and Randomized Leaky ReLu, shows a difference in the performance of the different models. This difference in performance is partially due to the fact that the model has been trained using the LeakyRelu activation function. This means that the model was not optimized for the two other activation functions, as this was not possible due to time and computational constraints. This gives the LeakyRelu a performance advantage over the other models, even though the activation functions are not very different from the trained on activation function. This performance difference is also seen in the reproduction results. A future research could be done on the different activation functions with training and inference evaluation with the same activation function in use. 
It is important to take into account that the authors probably did not choose LeakyRely for its performance, but for its specific characteristics. A relevant example would be that LeakyRelu is known for being able to compensate for Relu's dying neuron problem during training. The paper does not state proper reasons for the choice of activation function, so this might be another aspect to consider when retraining with different activation functions.

Addtionally, in the future the model could be trained on noisy data. Right now the model was trained on normal data and only inference and evaluation was done on noisy data. It would be interesting to see wether the model performs better on noisy data if it has also been trained on noisy data. We suspect that it will.


## Conclusion
The paper contains very interesting research and the results show great promise in the future of feature tracking. However, when attempting to reproduce the paper, it became apparent that the steps to reproduce the results need to be clarified and made more robust. 
The issues that arose during training make it impossible at this time to reproduce the results from scratch, which is important when validating the outcome of the paper. Once it becomes reproducible the paper can be validated by a group of peers and used as a basis for other research. 
Furthermore the inability to train the model on noisy data made it so that the only conclusion we can now come to is that the model does not show enough robustness. The immediate drop in performance at only 1% noise indicates a lack of performance in unknown situations, which is a crucial function of feature tracking.
## References

[1] Messikommer, N., Fang, C., Gehrig, M., & Scaramuzza, D. (2022). Data-driven Feature Tracking for Event Cameras. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2211.12826
      
[2] Multiflow Dataset (https://arxiv.org/abs/2203.13674)

[3] Event Camera dataset (EC) (https://arxiv.org/abs/1610.08336)

[4] Event-aided Direct Sparse Odometry dataset (EDS) (https://arxiv.org/abs/2204.07640)

[5] github (https://github.com/uzh-rpg/deep_ev_tracker/blob/master/models/correlation3_unscaled.py)

[6] Papers with Code - RReLU Explained. (n.d.). https://paperswithcode.com/method/rrelu




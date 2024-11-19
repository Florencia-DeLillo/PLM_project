# PLM_project
Project for PLM fineTuning

### TODO
- [ ] compute get_reps.py for both reps and logits
- [ ] fix the loss computation (cpu -> cuda)
- [ ] modify masking to output masking positions
- [ ] make sure masking is done always in the same way
- [ ] modify training loop
    - [ ] load pre-computed data
    - [ ] implement loss into data_utils.py
    - [ ] implement all functions with data_utils.py and their respective updates
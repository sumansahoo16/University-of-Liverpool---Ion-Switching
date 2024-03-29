# University-of-Liverpool---Ion-Switching


### Problem Statement

Ion channels are pore-forming membrane proteins, that allow ions to pass through the channel pore. When ion channels open, they pass electric currents.
Existing methods of detecting these state changes are slow and laborious.

![image](https://storage.googleapis.com/kaggle-media/competitions/Liverpool/ion%20image.jpg)

### Data
- A timestamp
- Ampere values (current) for each timestamp. These values were measured at a sampling frequency of 10 kHz.
- The corresponding number of open channels at that timestamp (only provided for the train data).

![image](https://user-images.githubusercontent.com/36400219/142971353-e8e5c21e-8ca9-44d9-bf0a-08eed260b90e.png)

### Pre-processing

We noticed that drift is present in the beginning and in the end of the signal data. We notice two types of drift: Slant drift and Parabolic drift. We removed them.

![image](https://www.kaggleusercontent.com/kf/29650685/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..heqe1kNIb9nop8MEVuggLg.Uud1a4t6fL_kTKzyFf7Hu1_NtCR5mxdi1LA7nEFcPAqM3TlJSAl29ZBCeJ2IEqRiE_X6P3xaKk545kImwWZcjh8gwZGOpFrrT4m0MOOtOgU2I3heQ6eAGXjKBOpd_qk_Oqklhtqvv6RgZgey2DZ79ZowTVc9CDacfaLv5qECQs9FhXvhKYhzDVVl0b7eRwobXie8EHaltdEp3RuGmQ0PFkyNg5xBlG1nmzOJeKQDZEI4Up1sHvvP3G9P4FO3uZUi9vki1kjzny6K_Qg7UW4Ruj_7-K-lSq9la63QFVeS41kmNSqOykyq6eYyYkWbBY_HnSEK1sqKP2y5JGamhaapVDGbF6GHmh5rcTvD_FQjo4aIKj2HjIMUvwkJQBWj3njkRVAwv1fXLYBWlwUtR8pWnaAmaHHbJEPEbDPcbzBTu1ybn7EKDSrcW7yN5cOUsIuKCkMQX7YI147SMr4czBkMbDzXdt1ExUu0nUxuez3oDKNzpc9RiG5aRarUfsa7bGPVXgBYw7Nl5Gbq2DARcywIppj1-BZepCI0lspiYuYkYUs-Z-HPRcW0Zm8x3IOcAvdZ7t-XgiYWy1j5pm2PKW2AlbSnWUYwaPjjTS4xrK9ENVmIXe0lETutu0napE10j0nh9BVk1YXGIaElcTbl82qzfkeO1NjE6H_gcJ7fJnX4njg.FHvLr02iD0xyBuMTJeYMwA/__results___files/__results___15_0.png)
![image](https://www.kaggleusercontent.com/kf/29650685/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..heqe1kNIb9nop8MEVuggLg.Uud1a4t6fL_kTKzyFf7Hu1_NtCR5mxdi1LA7nEFcPAqM3TlJSAl29ZBCeJ2IEqRiE_X6P3xaKk545kImwWZcjh8gwZGOpFrrT4m0MOOtOgU2I3heQ6eAGXjKBOpd_qk_Oqklhtqvv6RgZgey2DZ79ZowTVc9CDacfaLv5qECQs9FhXvhKYhzDVVl0b7eRwobXie8EHaltdEp3RuGmQ0PFkyNg5xBlG1nmzOJeKQDZEI4Up1sHvvP3G9P4FO3uZUi9vki1kjzny6K_Qg7UW4Ruj_7-K-lSq9la63QFVeS41kmNSqOykyq6eYyYkWbBY_HnSEK1sqKP2y5JGamhaapVDGbF6GHmh5rcTvD_FQjo4aIKj2HjIMUvwkJQBWj3njkRVAwv1fXLYBWlwUtR8pWnaAmaHHbJEPEbDPcbzBTu1ybn7EKDSrcW7yN5cOUsIuKCkMQX7YI147SMr4czBkMbDzXdt1ExUu0nUxuez3oDKNzpc9RiG5aRarUfsa7bGPVXgBYw7Nl5Gbq2DARcywIppj1-BZepCI0lspiYuYkYUs-Z-HPRcW0Zm8x3IOcAvdZ7t-XgiYWy1j5pm2PKW2AlbSnWUYwaPjjTS4xrK9ENVmIXe0lETutu0napE10j0nh9BVk1YXGIaElcTbl82qzfkeO1NjE6H_gcJ7fJnX4njg.FHvLr02iD0xyBuMTJeYMwA/__results___files/__results___15_1.png)
![image](https://www.kaggleusercontent.com/kf/29650685/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..heqe1kNIb9nop8MEVuggLg.Uud1a4t6fL_kTKzyFf7Hu1_NtCR5mxdi1LA7nEFcPAqM3TlJSAl29ZBCeJ2IEqRiE_X6P3xaKk545kImwWZcjh8gwZGOpFrrT4m0MOOtOgU2I3heQ6eAGXjKBOpd_qk_Oqklhtqvv6RgZgey2DZ79ZowTVc9CDacfaLv5qECQs9FhXvhKYhzDVVl0b7eRwobXie8EHaltdEp3RuGmQ0PFkyNg5xBlG1nmzOJeKQDZEI4Up1sHvvP3G9P4FO3uZUi9vki1kjzny6K_Qg7UW4Ruj_7-K-lSq9la63QFVeS41kmNSqOykyq6eYyYkWbBY_HnSEK1sqKP2y5JGamhaapVDGbF6GHmh5rcTvD_FQjo4aIKj2HjIMUvwkJQBWj3njkRVAwv1fXLYBWlwUtR8pWnaAmaHHbJEPEbDPcbzBTu1ybn7EKDSrcW7yN5cOUsIuKCkMQX7YI147SMr4czBkMbDzXdt1ExUu0nUxuez3oDKNzpc9RiG5aRarUfsa7bGPVXgBYw7Nl5Gbq2DARcywIppj1-BZepCI0lspiYuYkYUs-Z-HPRcW0Zm8x3IOcAvdZ7t-XgiYWy1j5pm2PKW2AlbSnWUYwaPjjTS4xrK9ENVmIXe0lETutu0napE10j0nh9BVk1YXGIaElcTbl82qzfkeO1NjE6H_gcJ7fJnX4njg.FHvLr02iD0xyBuMTJeYMwA/__results___files/__results___17_0.png)
![image](https://www.kaggleusercontent.com/kf/29650685/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..heqe1kNIb9nop8MEVuggLg.Uud1a4t6fL_kTKzyFf7Hu1_NtCR5mxdi1LA7nEFcPAqM3TlJSAl29ZBCeJ2IEqRiE_X6P3xaKk545kImwWZcjh8gwZGOpFrrT4m0MOOtOgU2I3heQ6eAGXjKBOpd_qk_Oqklhtqvv6RgZgey2DZ79ZowTVc9CDacfaLv5qECQs9FhXvhKYhzDVVl0b7eRwobXie8EHaltdEp3RuGmQ0PFkyNg5xBlG1nmzOJeKQDZEI4Up1sHvvP3G9P4FO3uZUi9vki1kjzny6K_Qg7UW4Ruj_7-K-lSq9la63QFVeS41kmNSqOykyq6eYyYkWbBY_HnSEK1sqKP2y5JGamhaapVDGbF6GHmh5rcTvD_FQjo4aIKj2HjIMUvwkJQBWj3njkRVAwv1fXLYBWlwUtR8pWnaAmaHHbJEPEbDPcbzBTu1ybn7EKDSrcW7yN5cOUsIuKCkMQX7YI147SMr4czBkMbDzXdt1ExUu0nUxuez3oDKNzpc9RiG5aRarUfsa7bGPVXgBYw7Nl5Gbq2DARcywIppj1-BZepCI0lspiYuYkYUs-Z-HPRcW0Zm8x3IOcAvdZ7t-XgiYWy1j5pm2PKW2AlbSnWUYwaPjjTS4xrK9ENVmIXe0lETutu0napE10j0nh9BVk1YXGIaElcTbl82qzfkeO1NjE6H_gcJ7fJnX4njg.FHvLr02iD0xyBuMTJeYMwA/__results___files/__results___17_1.png)

### Modeling

WaveNet is superior at finding patterns in waveforms. Using dilated convolutions, it decomposes a signal into its different frequency sine waves just like the Fourier Transform.

```
def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
        
        
    inp = Input(shape = (shape_))
    
    x = wave_block(x, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3, 1)
    
    out = Dense(11, activation = 'softmax', name = 'out')(x)
```


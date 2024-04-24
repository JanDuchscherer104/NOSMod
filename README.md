# pypho-ws

### Links
[**Root Raised Cosine Filter**](docs/FILTER.md)


### Questions

- In `sqrt_RNN_filter.py` `constpts_4qam` is not being used.
    ```py
    # Simulation
    # 4-QAM
    constpts_4qam = [(      [ 1.0*np.exp(2.0j*np.pi*x/4.0) for x in range(0,4)] ),
                    (      [(0),(0)], [(0),(1)], [(1),(1)], [(1),(0)]             )]
    ```

- In `sqrt_RNN_filter.py` the optical signal is being generated using np.random.uniform, an not using an electrical signal that is being converted into an optical signal.

- `Root Raised Cosine Filters` have a real frequency response, so there is no expected correlation between the real and imaginary parts of the signal. Same goes for both polarizations.

- Does it make sense to define  both the bandwidth and the samplig rate?

- How do the sampling rate and the symbol rate differ?



### Used Parameters
- `nos` = 128
- `sps` = 32
- `symbolrate` = 40e9
- `fwhm` = 0.33
- `B_filt` = 20
- `alpha` = 0.5
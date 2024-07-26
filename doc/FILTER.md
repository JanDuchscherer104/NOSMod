# Cosine Roll-off Filter

[**Latest PyTorch Implementation**](../src/nosmod/nosmod/sqrt_filter/raised_cosine_filter.py)

## Overview

The Cosine Roll-off filter, or [Raised-Cosine Filter (Wikipedia)](https://en.wikipedia.org/wiki/Raised-cosine_filter), is widely used in digital communication systems to control the bandwidth of a signal and reduce intersymbol interference (ISI). One of its primary purposes is to make the system more robust to timing errors and to optimize the usage of bandwidth. This filter is crucial in ensuring the transmitted signal maintains its integrity through the modulation and demodulation process.

<p float="left">
    <figure>
        <img src=".assets/raise_cos_freq_response.png" width="49%" />
        <img src=".assets/raise_cos_impulse_response.png" width="49%" />
        <figcaption>Frequency and Impulse Response of Raised Cosine Filter</figcaption>
    </figure>
</p>

$H(f=1/2f_s)=\frac{a_0}{2}$

## Theoretical Background

### Frequency Domain Behavior

The Raised-Cosine Filter achieves its roll-off characteristics by smoothly transitioning from the passband to the stopband using a cosine function. This gradual transition helps in minimizing the side lobes in the frequency spectrum, which in turn reduces ISI and bandwidth spillage into adjacent channels. The filter is designed to meet the [Nyquist ISI criterion (Wikipedia)](https://en.wikipedia.org/wiki/Nyquist_ISI_criterion), ensuring that the signal experiences zero ISI at the sampling instants.

### Mathematical Representation

The **frequency response** of the Raised Cosine filter in the transition band is defined as:

$$
H(f) = \begin{cases}
1, & |f| \leq \frac{1 - \beta}{2T} \\
\frac{1}{2} \left[1 + \cos \left( \frac{\pi T}{\beta} \left( |f| - \frac{1 - \beta}{2T} \right) \right) \right], & \frac{1 - \beta}{2T} < |f| \leq \frac{1 + \beta}{2T} \\
0, & \text{otherwise}
\end{cases}
$$

The **impulse response** of the Raised Cosine filter is given by:

$$
h(t) = \begin{cases}
\frac{\pi}{4T} \operatorname{sinc} \left( \frac{1}{2\beta} \right), & t = \pm \frac{T}{2\beta} \\
\frac{1}{T} \operatorname{sinc} \left( \frac{t}{T} \right) \frac{\cos \left( \frac{\pi \beta t}{T} \right)}{1 - \left( \frac{2\beta t}{T} \right)^2}, & \text{otherwise}
\end{cases}
$$

Where:
- $f$ is the frequency.
- $T$ is the symbol period.
- $\beta$ is the roll-off factor, ranging from 0 to 1, which controls the width of the filter's transition band.

### Roll-off Factor

The roll-off factor $\beta$ determines the excess bandwidth of the filter, i.e., the bandwidth occupied beyond the Nyquist bandwidth of $\frac{1}{2T}$. The value of $\beta$ ranges from 0 (no excess bandwidth, ideal brick-wall filter) to 1 (maximum excess bandwidth).

### Bandwidth

The bandwidth of a raised cosine filter is given by:

$$
B = \frac{f_s}{2} (\beta + 1)
$$

where $f_s = \frac{1}{T}$ is the symbol rate / sampling frequency.

### Application in Transmission Chains

In a typical digital communication system, the Raised-Cosine Filter is used both at the transmitter and the receiver to perform pulse shaping and matched filtering, respectively. This ensures that the overall response of the system meets the Nyquist criterion, thereby minimizing ISI and optimizing bandwidth usage. The filter's ability to smoothly transition between the passband and stopband makes it ideal for such applications, ensuring that narrowband filters can be employed without significant loss of information.

## Characteristics

- **`Df` (Deviation of Center Frequency)**: Adjusts the center frequency from the carrier, typically in GHz.
- **`B` (Bandwidth)**: Nominal bandwidth of the filter. The actual spectral width is affected by the roll-off factor.
- **`alpha` (Roll-off Factor)**: Defines the transition sharpness between the passband and stopband, ranging from 0 (sharp, rectangular) to 1 (gentle, maximal roll-off).
- **`loss`**: Attenuation in dB, representing energy loss due to filtering.

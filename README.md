# iGluSnFR-registration

This capsule does registration for iGluSnFR data.

# Usage:
```
python run_capsule.py \
    --input /path/to/tiffs \
    --output /results \
    --maxshift 50 \
    --ds_time 3 \
    --caiman_template True
```

# Outputs
```
results/simulation_description/
├── *_registered.tif          # Motion-corrected video
├── *_alignment_data.h5       # Shift vectors & metrics
└── *_channel_avg_8bit.tif    # Diagnostic average image
```

_alignment_data.h5 contains h5 file:
- `aData/numChannels`: The number of channels in the dataset.
- `aData/frametime`: The frame rate of the registered movie.
- `aData/motionR`: The values used to motion correct across the Y axis.
- `aData/motionC`: The values used to motion correct across the X axis.
- `aData/aError`: Alignment error.
- `aData/aRankCorr`: Rank correlation data.
- `aData/motionDSc`: The downsampled values used to motion correct across the X axis.
- `aData/motionDSr`: The downsampled values used to motion correct across the Y axis.
- `aData/recNegErr`: Negative reconstruction error metrics

```mermaid
flowchart TD
    A[Input Data] --> B[Downsample Time Series]
    B --> C[Calculate Correlation Matrix]
    C --> D[Hierarchical Clustering]
    D --> E[Find Best Template Frames]
    
    E --> F{Template Method}
    F -->|CaImAn| G1[Generate CaImAn Template]
    F -->|JNormCorre| G2[Generate JNormCorre Template]
    
    G1 --> H[Strip Registration]
    G2 --> H
    
    H --> I[Motion Estimation]
    I --> J[Interpolation]
    
    J --> K[Process Raw Frames]
    K --> L[Save Outputs]
    
    L --> M1[Registered Movie]
    L --> M2[Average Image]
    L --> M3[Alignment Data]
    
    subgraph Motion Estimation
        I1[DFT Registration] --> I2[Fast Cross-correlation]
        I2 --> I3[Template Update]
    end
    
    subgraph Outputs
        M1
        M2
        M3
    end

```

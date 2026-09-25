"""Published values used in the scattering comparison, in one place.

Every number is recorded, with its source and how it was read, in
data/gan_2dhg_measured.yaml.  Sheet densities, masses, Hall and quantum
mobilities are from Chang et al., Nat. Electron. 9, 346 (2026).

THE MEASURED LIFETIME RATIO
---------------------------
The ratio tau_tr / tau_q is compared with experiment as the ratio of the Hall
mobility to the quantum mobility.  Both mobilities are obtained from the data
without any effective mass (the Dingle factor is exp(-pi / mu_q B)), and
mu_tr / mu_q = tau_tr / tau_q whenever the same mass converts both.  This is
also the form in which the measurement paper states it (about 5 for the light
holes and 2 to 2.5 for the heavy holes).  Converting the two mobilities into
lifetimes separately is avoided: the light-hole quantum lifetime quoted in the
measurement paper, 0.15 ps, does not follow from its quantum mobility of
368 cm^2/Vs with its mass of 0.53 m0 (that gives 0.111 ps), so forming the
ratio from lifetimes would inherit an inconsistency that the mobility ratio
does not have.

Because the light-hole mass reported at high field (0.53 m0, averaged over
32 to 72 T) differs from its zero-field extrapolation (0.30 m0), the two
mobilities could in principle correspond to different masses: the Hall
mobility is measured below 9 T and the Dingle analysis at high field.  The
extreme assignment, zero-field mass for the Hall mobility and field-averaged
mass for the quantum mobility, lowers the light-hole ratio by the factor
0.30 / 0.53.  Comparisons are made against the ratio itself and against the
whole range this ambiguity allows.

The Hall mobilities are quoted as about 1900 and 400 cm^2/Vs, and the same
paper gives the range returned by the fit, 1858 to 1986 and 381 to 464 cm^2/Vs
at 3 K.  The nominal ratios use the rounded values; the uncertainty ranges and
the permissive box include the fit ranges.
"""

N_L_CM2 = 0.80e13          # light pair, from the 166 T oscillation frequency
N_H_CM2 = 3.80e13          # heavy pair, from the 795 T oscillation frequency
M_L = 0.53                 # field-averaged light mass, 32 to 72 T
M_L_ZERO_FIELD = 0.30      # linear extrapolation of the same data to B = 0
M_H = 1.92

MU_HALL_L = 1900.0         # cm^2/Vs, 3 K, two-carrier fit to 9 T
MU_HALL_H = 400.0
MU_HALL_L_RANGE = (1858.0, 1986.0)   # range of the same fit, as reported
MU_HALL_H_RANGE = (381.0, 464.0)
MU_Q_L = 368.0             # +/- 14, Dingle analysis
MU_Q_L_ERR = 14.0
MU_Q_H_RANGE = (167.0, 200.0)   # from the onset of the heavy oscillations

# Nominal ratios, from the rounded Hall mobilities (1900 and 400).
R_L = MU_HALL_L / MU_Q_L                                   # 5.16
R_H_NOMINAL_RANGE = (MU_HALL_H / MU_Q_H_RANGE[1],
                     MU_HALL_H / MU_Q_H_RANGE[0])          # 2.00 to 2.40
R_H = 0.5 * (R_H_NOMINAL_RANGE[0] + R_H_NOMINAL_RANGE[1])  # 2.20

# Uncertainty ranges, combining the reported range of each Hall mobility with
# the uncertainty or range of the corresponding quantum mobility.
R_L_RANGE = (MU_HALL_L_RANGE[0] / (MU_Q_L + MU_Q_L_ERR),
             MU_HALL_L_RANGE[1] / (MU_Q_L - MU_Q_L_ERR))   # 4.86 to 5.61
R_H_RANGE = (MU_HALL_H_RANGE[0] / MU_Q_H_RANGE[1],
             MU_HALL_H_RANGE[1] / MU_Q_H_RANGE[0])         # 1.91 to 2.78
R_L_LOW = R_L_RANGE[0] * M_L_ZERO_FIELD / M_L              # 2.75

TARGET_POINT = ((R_L, R_L), (R_H, R_H))
TARGET_BOX = ((R_L_LOW, R_L_RANGE[1]), R_H_RANGE)

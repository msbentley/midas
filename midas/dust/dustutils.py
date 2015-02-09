def dmeanmass(m0, m1, f0, f1):

    q = f1/f0
    dmeanmass = m0 * dkappa( q*m1/m0 ) / dkappa( q )

    return dmeanmass	

# FUNCTION DMEANMASS, M0, M1, F0, F1
# 
# ; CALCULATES THE MEAN MASS IN A MASS INTERVAL
# ; ASSUMING THAT THE MASS DISTRIBUTION IS POWER LAW
# 
# ; I M0,M1    R*8 BOUNDARY OF MASS INTERVAL
# ; I F0,F1    R*8 VALUES OF THE CUMULATIVE MASS DISTRIBUTION AT
# ; I              THE INTERVAL BOUNDARIES
# ;
# ; O MEANMASS R*8 MEAN MASS IN THE MASS INTERVAL
# 
#     Q = F1/F0
#     DMEANMASS = M0 * DKAPPA( Q*M1/M0 ) / DKAPPA( Q )
# 
#     RETURN, DMEANMASS
# 
# END

def dkappa(x):

    import math

    if (x == 1.):
        dkappa = 1.
    else:
#         dkappa = (x-1.)/10.**(x)
        dkappa = (x-1.)/math.log(x)
    return dkappa

# FUNCTION DKAPPA, X
# 
# ; COMPUTES THE KAPPA FUNCTION
# ; KAPPA(X)=INT {V=0..1} X**V DV
# ;         =1,           X=1
# ;         =(X-1)/LN(X), OTHERWISE
# ;
# ; I X    R*8 X>0
# 
#     IF (X EQ 1) THEN BEGIN
#        DKAPPA=1
#     ENDIF ELSE BEGIN
#        DKAPPA=(X-1.0D)/ALOG(X)
#     END
#     RETURN, DKAPPA
# 
# END

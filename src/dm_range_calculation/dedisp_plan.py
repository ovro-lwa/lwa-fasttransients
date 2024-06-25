from numpy import *
import argparse
import json

class observation:
    def __init__(self, dt, f_ctr, BW, numchan, cDM):
        # dt in sec, f_ctr and in MHz
        self.dt = dt
        self.f_ctr = f_ctr
        self.BW = BW
        self.numchan = numchan
        self.chanwidth = BW/numchan
        self.cDM = cDM

    def guess_dDM(self, DM):
        """
        guess_dDM(self, DM):
            Choose a reasonable dDM by setting the maximum smearing across the
                'BW' to equal the sampling time 'dt'.
        """
        return self.dt*0.0001205*self.f_ctr**3.0/(0.5*self.BW)

class dedisp_method:
    def __init__(self, obs, downsamp, loDM, hiDM, dDM, numDMs=0, numsub=0, smearfact=2.0):
        self.obs = obs
        self.downsamp = downsamp
        self.loDM = loDM
        self.dDM = dDM
        self.numsub = numsub
        self.BW_smearing = BW_smear(dDM, self.obs.BW, self.obs.f_ctr)
        self.numprepsub = 0

        if numsub:  # Calculate the maximum subband smearing we can handle
            DMs_per_prepsub = 2
            while True:
                next_dsubDM = (DMs_per_prepsub+2) * dDM
                next_ss = subband_smear(next_dsubDM, numsub, self.obs.BW, self.obs.f_ctr)
                # The 0.8 is a small fudge factor to make sure that the subband
                # smearing is always the smallest contribution
                if next_ss > 0.8 * min(self.BW_smearing, 1000.0 * obs.dt * downsamp):
                    self.dsubDM = DMs_per_prepsub * dDM
                    self.DMs_per_prepsub = DMs_per_prepsub
                    self.sub_smearing = subband_smear(self.dsubDM, numsub, self.obs.BW, self.obs.f_ctr)
                    break
                DMs_per_prepsub += 2
        else:
            self.dsubDM = dDM
            self.sub_smearing = subband_smear(self.dsubDM, numsub, self.obs.BW, self.obs.f_ctr)
        # Calculate the nominal DM to move to the next method
        cross_DM = self.DM_for_smearfact(smearfact)
        if cross_DM > hiDM:
            cross_DM = hiDM
        if numDMs == 0:
            self.numDMs = int(ceil((cross_DM-loDM)/dDM))
            if numsub:
                self.numprepsub = int(ceil(self.numDMs * dDM / self.dsubDM))
                self.numDMs = self.numprepsub * DMs_per_prepsub
        else:
            self.numDMs = numDMs
        self.hiDM = loDM + self.numDMs * dDM
        self.DMs = arange(self.numDMs, dtype='d') * dDM + loDM

    def chan_smear(self, DM):
        """
        Return the smearing (in ms) in each channel at the specified DM
        """
        try:
            DM = where(DM-self.obs.cDM == 0.0, self.obs.cDM + self.dDM/2.0, DM)
        except TypeError:
            if DM - self.obs.cDM == 0.0:
                DM = self.obs.cDM + self.dDM/2.0
        return dm_smear(DM, self.obs.chanwidth, self.obs.f_ctr, self.obs.cDM)

    def total_smear(self, DM):
        """
        Return the total smearing in ms due to the sampling rate,
        the smearing over each channel, the smearing over each subband
        (if numsub > 0) and the smearing over the full BW assuming the
        worst-case DM error.
        """
        return sqrt((1000.0 * self.obs.dt)**2.0 +
                    (1000.0 * self.obs.dt * self.downsamp)**2.0 +
                    self.BW_smearing**2.0 +
                    self.sub_smearing**2.0 +
                    self.chan_smear(DM)**2.0)

    def DM_for_smearfact(self, smearfact):
        """
        Return the DM where the smearing in a single channel is a factor smearfact
        larger than all the other smearing causes combined.
        """
        other_smear = sqrt((1000.0 * self.obs.dt)**2.0 +
                           (1000.0 * self.obs.dt * self.downsamp)**2.0 +
                           self.BW_smearing**2.0 +
                           self.sub_smearing**2.0)
        return smearfact * 0.001 * other_smear / self.obs.chanwidth * 0.0001205 * self.obs.f_ctr**3.0 + self.obs.cDM

    def DM_for_newparams(self, dDM, downsamp):
        """
        Return the DM where the smearing in a single channel is causes the same smearing
        as the effects of the new downsampling rate and dDM.
        """
        other_smear = sqrt((1000.0 * self.obs.dt)**2.0 +
                           (1000.0 * self.obs.dt * downsamp)**2.0 +
                           BW_smear(dDM, self.obs.BW, self.obs.f_ctr)**2.0 +
                           self.sub_smearing**2.0)
        return 0.001 * other_smear / self.obs.chanwidth * 0.0001205 * self.obs.f_ctr**3.0

    def __str__(self):
        if self.numsub:
            return "%9.3f  %9.3f  %6.2f    %4d  %6.2f  %6d  %6d  %6d " % \
                   (self.loDM, self.hiDM, self.dDM, self.downsamp, self.dsubDM,
                    self.numDMs, self.DMs_per_prepsub, self.numprepsub)
        else:
            return "%9.3f  %9.3f  %6.2f    %4d  %6d" %  \
                   (self.loDM, self.hiDM, self.dDM, self.downsamp, self.numDMs)

def dm_smear(DM, BW, f_ctr, cDM=0.0):
    """
    dm_smear(DM, BW, f_ctr, cDM=0.0):
        Return the smearing in ms caused by a 'DM' over a bandwidth
        of 'BW' MHz centered at 'f_ctr' MHz.
    """
    return 1000.0 * fabs(DM-cDM) * BW / (0.0001205 * f_ctr**3.0)

def BW_smear(DMstep, BW, f_ctr):
    """
    BW_smear(DMstep, BW, f_ctr):
        Return the smearing in ms caused by a search using a DM stepsize of
        'DMstep' over a bandwidth of 'BW' MHz centered at 'f_ctr' MHz.
    """
    maxDMerror = 0.5 * DMstep
    return dm_smear(maxDMerror, BW, f_ctr)

def guess_DMstep(dt, BW, f_ctr):
    """
    guess_DMstep(dt, BW, f_ctr):
        Choose a reasonable DMstep by setting the maximum smearing across the
        'BW' to equal the sampling time 'dt'.
    """
    return dt * 0.0001205 * f_ctr**3.0 / (0.5 * BW)

def subband_smear(subDMstep, numsub, BW, f_ctr):
    """
    subband_smear(subDMstep, numsub, BW, f_ctr):
        Return the smearing in ms caused by a search using a subband
        DM stepsize of 'subDMstep' over a total bandwidth of 'BW' MHz
        centered at 'f_ctr' MHz, and having numsub subbands.
    """
    if numsub == 0:
        return 0.0
    subBW = BW / numsub
    maxsubDMerror = 0.5 * subDMstep
    return dm_smear(maxsubDMerror, subBW, f_ctr)

def total_smear(DM, DMstep, dt, f_ctr, BW, numchan, subDMstep, cohdm=0.0, numsub=0):
    """
    total_smear(DM, DMstep, dt, f_ctr, BW, numchan, subDMstep, cohdm=0.0, numsub=0):
        Return the total smearing in ms due to the sampling rate,
        the smearing over each channel, the smearing over each subband
        (if numsub > 0) and the smearing over the full BW assuming the
        worst-case DM error.
    """
    return sqrt(2 * (1000.0 * dt)**2.0 +
                dm_smear(DM, BW / numchan, f_ctr, cohdm)**2.0 +
                subband_smear(subDMstep, numsub, BW, f_ctr)**2.0 +
                BW_smear(DMstep, BW, f_ctr)**2.0)

def dm_steps(loDM, hiDM, obs, cohdm=0.0, numsub=0, ok_smearing=0.0):
    """
    dm_steps(loDM, hiDM, obs, cohdm=0.0, numsub=0, ok_smearing=0.0):
        Return the optimal DM stepsizes (and subband DM stepsizes if
        numsub>0) to keep the total smearing below 'ok_smearing' (in ms),
        for the DMs between loDM and hiDM.  If 'ok_smearing'=0.0, then
        use the best values based only on the data.
    """
    # Allowable DM stepsizes
    allow_dDMs = [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0,
                  2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0]
    # Allowable number of downsampling factors
    allow_downsamps = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Initial values
    index_downsamps = index_dDMs = 0
    downsamp = allow_downsamps[index_downsamps]
    dDM = allow_dDMs[index_dDMs]
    dtms = 1000.0 * obs.dt

    # Fudge factor that "softens" the boundary defining
    # if 2 time scales are equal or not
    ff = 1.2

    # This is the array that will hold the de-dispersion plans
    methods = []

    # Minimum possible smearing
    min_tot_smearing = total_smear(loDM + 0.5 * dDM, dDM, obs.dt, obs.f_ctr, obs.BW, obs.numchan, allow_dDMs[0], cohdm, 0)
    # Minimum channel smearing
    min_chan_smearing = dm_smear(linspace(loDM, hiDM, 10000), obs.chanwidth, obs.f_ctr, cohdm).min()
    # Minimum smearing across the obs.BW
    min_BW_smearing = BW_smear(dDM, obs.BW, obs.f_ctr)

    print()
    print("Minimum total smearing     : %.3g ms" % min_tot_smearing)
    print("--------------------------------------------")
    print("Minimum channel smearing   : %.3g ms" % min_chan_smearing)
    print("Minimum smearing across BW : %.3g ms" % min_BW_smearing)
    print("Minimum sample time        : %.3g ms" % dtms)
    print()

    ok_smearing = max([ok_smearing, min_chan_smearing, min_BW_smearing, dtms])
    print("Setting the new 'best' resolution to : %.3g ms" % ok_smearing)

    # See if the data is too high time resolution for our needs
    if ff * min_chan_smearing > dtms or ok_smearing > dtms:
        if ok_smearing > ff * min_chan_smearing:
            print("   Note: ok_smearing > dt (i.e. data is higher resolution than needed)")
            okval = ok_smearing
        else:
            print("   Note: min_chan_smearing > dt (i.e. data is higher resolution than needed)")
            okval = ff * min_chan_smearing

        while index_downsamps + 1 < len(allow_downsamps) and dtms * allow_downsamps[index_downsamps + 1] < okval:
            index_downsamps += 1
        downsamp = allow_downsamps[index_downsamps]

        print("         New dt is %d x %.12g ms = %.12g ms" %
              (downsamp, dtms, dtms * downsamp))

    # Calculate the appropriate initial dDM
    dDM = guess_DMstep(obs.dt * downsamp, obs.BW, obs.f_ctr)
    print("Best guess for optimal initial dDM is %.3f" % dDM)
    while allow_dDMs[index_dDMs + 1] < ff * dDM:
        index_dDMs += 1

    # Create the first method
    methods = [dedisp_method(obs, downsamp, loDM, hiDM, allow_dDMs[index_dDMs], numsub=numsub)]
    numDMs = [methods[-1].numDMs]

    # Calculate the next methods
    while methods[-1].hiDM < hiDM:
        # Determine the new downsample factor
        index_downsamps += 1
        downsamp = allow_downsamps[index_downsamps]
        eff_dt = dtms * downsamp

        # Determine the new DM step
        while BW_smear(allow_dDMs[index_dDMs + 1], obs.BW, obs.f_ctr) < ff * eff_dt:
            index_dDMs += 1
        dDM = allow_dDMs[index_dDMs]

        # Get the next method
        methods.append(dedisp_method(obs, downsamp, methods[-1].hiDM, hiDM, dDM, numsub=numsub))
        numDMs.append(methods[-1].numDMs)

    # Calculate the DMs to search and the smearing at each
    total_numDMs = sum(numDMs)
    DMs = zeros(total_numDMs, dtype='d')
    total_smears = zeros(total_numDMs, dtype='d')

    # Calculate the DMs and optimal smearing for all the DMs
    for ii, offset in enumerate(add.accumulate([0] + numDMs[:-1])):
        DMs[offset:offset + numDMs[ii]] = methods[ii].DMs
        total_smears[offset:offset + numDMs[ii]] = methods[ii].total_smear(methods[ii].DMs)

    # Calculate the predicted amount of time that will be spent in searching
    # this batch of DMs as a fraction of the total
    work_fracts = [meth.numDMs / float(meth.downsamp) for meth in methods]
    work_fracts = asarray(work_fracts) / sum(work_fracts)

    results = []
    # The optimal smearing
    tot_smear = total_smear(DMs, allow_dDMs[0], obs.dt, obs.f_ctr, obs.BW, obs.numchan, allow_dDMs[0], cohdm, 0)

    if numsub:
        print("\n  Low DM    High DM     dDM  DownSamp  dsubDM   #DMs  DMs/call  calls  WorkFract")
    else:
        print("\n  Low DM    High DM     dDM  DownSamp   #DMs  WorkFract")
    for method, fract in zip(methods, work_fracts):
        if numsub:
            result = {
                "Low DM": method.loDM,
                "High DM": method.hiDM,
                "dDM": method.dDM,
                "DownSamp": method.downsamp,
                "dsubDM": method.dsubDM,
                "#DMs": method.numDMs,
                "DMs/call": method.DMs_per_prepsub,
                "calls": method.numprepsub,
                "WorkFract": fract
            }
            results.append(result)
            print(method, "  %.4g" % fract)
        else:
            result = {
                "Low DM": method.loDM,
                "High DM": method.hiDM,
                "dDM": method.dDM,
                "DownSamp": method.downsamp,
                "#DMs": method.numDMs,
                "WorkFract": fract
            }
            results.append(result)
            print(method, "  %.4g" % fract)

    return methods, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a good plan for de-dispersing data.')

    parser.add_argument('-l', '--loDM', type=float, default=0.0, help='Low DM to search (default = 0 pc cm-3)')
    parser.add_argument('-d', '--hiDM', type=float, default=1000.0, help='High DM to search (default = 1000 pc cm-3)')
    parser.add_argument('-f', '--fctr', type=float, default=1400.0, help='Center frequency (default = 1400MHz)')
    parser.add_argument('-b', '--bw', type=float, default=300.0, help='Bandwidth in MHz (default = 300MHz)')
    parser.add_argument('-n', '--numchan', type=int, default=1024, help='Number of channels (default = 1024)')
    parser.add_argument('-c', '--cohdm', type=float, default=0.0, help='Coherent DM in each chan (default = 0.0)')
    parser.add_argument('-t', '--dt', type=float, default=0.000064, help='Sample time (s) (default = 0.000064 s)')
    parser.add_argument('-s', '--subbands', type=int, default=0, help='Number of subbands (default = #chan)')
    parser.add_argument('-r', '--res', type=float, default=0.0, help='Acceptable time resolution (ms)')
    parser.add_argument('--output', type=str, default='dm_ranges.json', help='Output JSON file for DM ranges')

    args = parser.parse_args()

    # The following is an instance of an "observation" class
    obs = observation(args.dt, args.fctr, args.bw, args.numchan, args.cohdm)
    # The following function creates the de-dispersion plan
    # The args.res values is optional and allows you to raise the floor
    # and provide a level of smearing that you are willing to accept (in ms)
    methods, results = dm_steps(args.loDM, args.hiDM, obs, args.cohdm, args.subbands, args.res)
    
    with open(args.output, 'w') as outfile:
        json.dump(results, outfile, indent=4)

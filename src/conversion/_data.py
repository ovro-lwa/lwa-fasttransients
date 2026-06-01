"""
Module to help with manipulating HDF5 beam data files.
"""

import os
import h5py
import numpy
import warnings
from datetime import datetime, timezone
from collections import defaultdict

from lsl.common import ndp, mcs, sdf, metabundle
from lsl.reader.drx import FILTER_CODES


__version__ = "1.1"
__all__ = ['create_new_file', 'fill_minimum', 'fill_from_metabundle', 'fill_from_sdf', 
           'get_observation_set', 'create_observation_set', 'get_time', 'get_data_set']



HDF5_CHUNK_SIZE_MB = 32


def _valuetoDelay(value):
    try:
        return mcs.mcsd_to_delay(value)
    except:
        value = ((value & 0xFF) << 8) | ((value >>8) & 0xFF)
        return ndp.ndpd_to_delay(value)


def _valuetoGain(value):
    try:
        return mcs.mcsg_to_gain(value)
    except:
        value = ((value & 0xFF) << 8) | ((value >>8) & 0xFF)
        return ndp.ndpg_to_gain(value)


class _HDFFileRegistry(object):
    """
    Class to keep track of which HDF files are open so that we can close them 
    out when we exit.
    
    This concept/framework/class is borrowed from PyTables:
        https://github.com/PyTables/PyTables/blob/master/tables/file.py
    """
    
    def __init__(self):
        self._name_mapping = defaultdict(set)
        self._handlers = set()
        
    @property
    def filenames(self):
        return list(self._name_mapping.keys())
        
    @property
    def handlers(self):
        return self._handlers
        
    def __len__(self):
        return len(self._handlers)
        
    def __contains__(self, filename):
        return filename in self.filenames
        
    def add(self, handler):
        self._name_mapping[handler.alt_filename].add(handler)
        self._handlers.add(handler)
        
    def remove(self, handler):
        filename = handler.alt_filename
        self._name_mapping[filename].remove(handler)
        if not self._name_mapping[filename]:
            del self._name_mapping[filename]
        self._handlers.remove(handler)
        
    def close_all(self):
        handlers = list(self.handlers)  # make a copy
        for handler in handlers:
            handler.close()


_open_hdf_files = _HDFFileRegistry()


class HDFFileWrapper(h5py.File):
    """
    Sub-class of h5py.File that allows us to hook in to the HDF file registry 
    so that any open HDF files are closed automatically on exit.
    """
    
    def __init__(self, name, mode=None, driver=None, libver=None, userblock_size=None, **kwds):
        super(HDFFileWrapper, self).__init__(name, mode=mode, driver=driver, libver=libver, userblock_size=userblock_size, **kwds)
        self.alt_filename = self.filename  # We need to store this because it
                                           # disappears when close() is called
        _open_hdf_files.add(self)
        
    def close(self):
        super(HDFFileWrapper, self).close()
        _open_hdf_files.remove(self)


def create_new_file(filename):
    """
    Create a new HDF5 and return the handle for it.  This sets up all of 
    the required attributes and groups and fills them with dummy values.
    
    Returns an open h5py.File instance.
    """
    
    # Create the file
    f = HDFFileWrapper(filename, 'w')
    
    # Observer and Project Info.
    f.attrs['ObserverID'] = 0
    f.attrs['ObserverName'] = ''
    f.attrs['ProjectID'] = ''
    f.attrs['SessionID'] = 0
    
    # Station information
    f.attrs['StationName'] = ''
    
    # File creation time
    f.attrs['FileCreation'] = datetime.now(timezone.utc).strftime("UTC %Y/%m/%d %H:%M:%S")
    f.attrs['FileGenerator'] = ''
    
    # Input file info.
    f.attrs['InputData'] = ''
    f.attrs['InputMetadata'] = ''
    
    return f


def fill_minimum(f, obsID, beam, srate, srateUnits='samples/s', station=None):
    """
    Minimum metadata filling for a particular observation.
    """
    
    # Station information
    if station is not None:
        if station in ('lwa1', 'lwasv', 'lwana', 'lwacs', 'lwamc', 'lwabb', 'ovrolwa'):
            f.attrs['StationName'] = station
        else:
            raise ValueError("Unknown station name: %s" % station)
            
    # Get the group or create it if it doesn't exist
    obs = f.get('/Observation%i' % obsID, None)
    if obs is None:
        obs = f.create_group('/Observation%i' % obsID)
        
    # Target info.
    obs.attrs['TargetName'] = ''
    obs.attrs['RA'] = -99.0
    obs.attrs['RA_Units'] = 'hours'
    obs.attrs['Dec'] = -99.0
    obs.attrs['Dec_Units'] = 'degrees'
    obs.attrs['Epoch'] = 2000.0
    obs.attrs['TrackingMode'] = 'Unknown'
    
    # Observation info
    obs.attrs['ARX_Filter'] = -1.0
    obs.attrs['ARX_Gain1'] = -1.0
    obs.attrs['ARX_Gain2'] = -1.0
    obs.attrs['ARX_GainS'] = -1.0
    obs.attrs['Beam'] = beam
    obs.attrs['DRX_Gain'] = -1.0
    obs.attrs['sampleRate'] = srate
    obs.attrs['sampleRate_Units'] = srateUnits
    obs.attrs['tInt'] = -1.0
    obs.attrs['tInt_Units'] = 's'
    obs.attrs['LFFT'] = -1
    obs.attrs['nChan'] = -1
    obs.attrs['RBW'] = -1.0
    obs.attrs['RBW_Units'] = 'Hz'
    
    return True


def fill_from_metabundle(f, tarball):
    """
    Fill in a HDF5 file based off an input metadata file.
    """
    
    # Pull out what we need from the tarball
    project = metabundle.get_sdf(tarball)
    cds = metabundle.get_command_script(tarball)
    station = metabundle.get_mcs_hostname(tarball)
    station = station.split('-', 1)[0]
    
    # Observer and Project Info.
    f.attrs['ObserverID'] = project.observer.id
    f.attrs['ObserverName'] = project.observer.name
    f.attrs['ProjectID'] = project.id
    f.attrs['SessionID'] = project.sessions[0].id
    
    # Station information
    f.attrs['StationName'] = station
    
    # Input file info.
    f.attrs['InputMetadata'] = os.path.basename(tarball)
    
    # ARX configuration summary
    try:
        arx = metabundle.get_asp_configuration_summary(tarball)
    except:
        arx = {'asp_filter': -1, 'asp_atten_1': -1, 'asp_atten_2': -1, 'asp_atten_3': -1}
        
    for i,obsS in enumerate(project.sessions[0].observations):
        # Detailed observation information
        try:
            obsD = metabundle.get_observation_spec(tarball, obs_id=i+1)
        except OSError:
            ## This currently fails on .tgz files from LWA-NA.  Switch over to
            ## SDF mode
            return fill_from_sdf(f, project, station=station)
            
        # Get the group or create it if it doesn't exist
        grp = f.get('/Observation%i' % (i+1,), None)
        if grp is None:
            grp = f.create_group('/Observation%i' % (i+1))
            
        # Target info.
        grp.attrs['ObservationName'] = obsS.name
        grp.attrs['TargetName'] = obsS.target
        grp.attrs['RA'] = obsD['ra']
        grp.attrs['RA_Units'] = 'hours'
        grp.attrs['Dec'] = obsD['dec']
        grp.attrs['Dec_Units'] = 'degrees'
        grp.attrs['Epoch'] = 2000.0
        grp.attrs['TrackingMode'] = obsD['mode'].name
        
        # Observation info
        grp.attrs['ARX_Filter'] = arx['asp_filter']
        grp.attrs['ARX_Gain1'] = arx['asp_atten_1']
        grp.attrs['ARX_Gain2'] = arx['asp_atten_2']
        grp.attrs['ARX_Gain3'] = arx['asp_atten_3']
        grp.attrs['Beam'] = obsD['drx_beam']
        grp.attrs['DRX_Gain'] = obsD['drx_gain']
        grp.attrs['sampleRate'] = float(FILTER_CODES[obsD['bw']])
        grp.attrs['sampleRate_Units'] = 'samples/s'
        
        # Deal with stepped mode
        if obsD['mode'].name == 'STEPPED':
            stps = grp.create_group('Pointing')
            stps.attrs['StepType'] = 'RA/Dec' if obsD['is_radec'] else 'Az/Alt'
            stps.attrs['col0'] = 'StartTime'
            stps.attrs['col0_Unit'] = 's'
            stps.attrs['col1'] = 'RA' if obsD['is_radec'] else 'Azimuth'
            stps.attrs['col1_Unit'] = 'h' if obsD['is_radec'] else 'd'
            stps.attrs['col2'] = 'Dec' if obsD['is_radec'] else 'Elevation'
            stps.attrs['col2_Unit'] = 'd'
            stps.attrs['col3'] = 'Tuning1'
            stps.attrs['col3_Unit'] = 'Hz'
            stps.attrs['col4'] = 'Tuning2'
            stps.attrs['col4_Unit'] = 'Hz'
        
            # Extract the data for the steps
            data = numpy.zeros((len(obsD['steps']), 5))
            t = obsD['mjd']*86400.0 + obsD['mpm']/1000.0 - 3506716800.0
            for i,s in enumerate(obsD['steps']):
                data[i,0] = t
                data[i,1] = s.OBS_STP_C1
                data[i,2] = s.OBS_STP_C2
                data[i,3] = ndp.word_to_freq(s.OBS_STP_FREQ1)
                data[i,4] = ndp.word_to_freq(s.OBS_STP_FREQ2)
            
                ## Update the start time for the next step
                t += s.OBS_STP_T / 1000.0
                
            # Save it
            stps['Steps'] = data
                
            # Deal with specified delays and gains if needed
            if obsD['steps'][0].OBS_STP_B == 3:
                nstand = 256
                label_base = 'NDP'
                if station == 'ovrolwa':
                    nstand = 352
                    label_base = 'OVR'
                    
                cbfg = grp.create_group('CustomBeamforming')
                dlys = cbfg.create_dataset('Delays', (len(obsD['steps']), nstand*2+1), 'f4')
                dlys.attrs['col0'] = 'StartTime'
                dlys.attrs['col0_Unit'] = 's'
                for j in range(2*nstand):
                    dlys.attrs['col%i' % (j+1)] = '%s Digitizer %i' % (label_base, j+1)
                    dlys.attrs['col%i_Unit' % (j+1)] = 'ns'
                    
                # Extract the delays
                dataD = numpy.zeros((len(obsD['steps']), 2*nstand+1))
                t = obsD['mjd']*86400.0 + obsD['mpm']/1000.0 - 3506716800.0
                for i,s in enumerate(obsD['steps']):
                    dataD[i,0] = t
                    for j in range(2*nstand):
                        dataD[i,1+j] = _valuetoDelay(s.delay[j])
                        
                # Save the delays
                dlys[:,:] = dataD
                
                gais = cbfg.create_dataset('Gains', (len(obsD['steps']), nstand*2*2+1), 'f4')
                gais.attrs['col0'] = 'StartTime'
                gais.attrs['col0_Unit'] = 's'
                m = 1
                for j in range(nstand):
                    for k in range(2):
                        for l in range(2):
                            gais.attrs['col%i' % m] = '%s Stand %i %s contribution to beam %s' % (label_base, j+1, 'X' if k == 0 else 'Y', 'X' if l == 0 else 'Y')
                            gais.attrs['col%i_Unit' % m] = 'None'
                            m += 1
                            
                # Extract the gains
                dataG = numpy.zeros((len(obsD['steps']), nstand*2*2+1))
                for i,s in enumerate(obsD['steps']):
                    dataG[i,0] = t
                    for j in range(nstand):
                        dataG[i,1+4*j+0] = _valuetoGain(s.gain[j][0][0])
                        dataG[i,1+4*j+1] = _valuetoGain(s.gain[j][0][1])
                        dataG[i,1+4*j+2] = _valuetoGain(s.gain[j][1][0])
                        dataG[i,1+4*j+3] = _valuetoGain(s.gain[j][1][1])
                         
                # Save the gains
                gais[:,:] = dataG
                
    return True


def fill_from_sdf(f, sdf_or_sdfFilename, station=None):
    """
    Fill in a HDF5 file based off an input session definition file.
    """
    
    # Pull out what we need from the tarball or maybe we are lucky and we
    # already have what we need
    if isinstance(sdf_or_sdfFilename, sdf.Project):
        project = sdf_or_sdfFilename
        sdf_or_sdfFilename = 'preparsed_sdf.info'
    else:
        project = sdf.parse_sdf(sdf_or_sdfFilename)
        
    # Observer and Project Info.
    f.attrs['ObserverID'] = project.observer.id
    f.attrs['ObserverName'] = project.observer.name
    f.attrs['ProjectID'] = project.id
    f.attrs['SessionID'] = project.sessions[0].id
    
    # Station information
    if station is not None:
        if station in ('lwa1', 'lwasv', 'lwana', 'lwacs', 'lwamc', 'lwabb', 'ovrolwa'):
            f.attrs['StationName'] = station
        else:
            raise ValueError("Unknown station name: %s" % station)
            
    # Input file info.
    f.attrs['InputMetadata'] = os.path.basename(sdf_or_sdfFilename)
    
    # ARX configuration summary
    arx = {'asp_filter': -1, 'asp_atten_1': -1, 'asp_atten_2': -1, 'asp_atten_3': -1}
    arx['asp_filter'] = numpy.median( project.sessions[0].observations[0].asp_filter )
    arx['asp_atten_1'] = numpy.median( project.sessions[0].observations[0].asp_atten_1 )
    arx['asp_atten_2'] = numpy.median( project.sessions[0].observations[0].asp_atten_2 )
    arx['asp_atten_3'] = numpy.median( project.sessions[0].observations[0].asp_atten_3 )
    
    for i,obsS in enumerate(project.sessions[0].observations):
        # Get the group or create it if it doesn't exist
        grp = f.get('/Observation%i' % (i+1,), None)
        if grp is None:
            grp = f.create_group('/Observation%i' % (i+1))
            
        # Target info.
        grp.attrs['ObservationName'] = obsS.name
        grp.attrs['TargetName'] = obsS.target
        grp.attrs['RA'] = obsS.ra
        grp.attrs['RA_Units'] = 'hours'
        grp.attrs['Dec'] = obsS.dec
        grp.attrs['Dec_Units'] = 'degrees'
        grp.attrs['Epoch'] = 2000.0
        grp.attrs['TrackingMode'] = obsS.mode
        
        # Observation info
        grp.attrs['ARX_Filter'] = arx['asp_filter']
        grp.attrs['ARX_Gain1'] = arx['asp_atten_1']
        grp.attrs['ARX_Gain2'] = arx['asp_atten_2']
        grp.attrs['ARX_Gain3'] = arx['asp_atten_3']
        grp.attrs['Beam'] = project.sessions[0].drx_beam
        grp.attrs['DRX_Gain'] = obsS.gain
        grp.attrs['sampleRate'] = float(FILTER_CODES[obsS.filter])
        grp.attrs['sampleRate_Units'] = 'samples/s'
        
        # Deal with stepped mode
        if obsS.mode == 'STEPPED':
            stps = grp.create_group('Pointing')
            stps.attrs['StepType'] = 'RA/Dec' if obsS.is_radec else 'Az/Alt'
            stps.attrs['col0'] = 'StartTime'
            stps.attrs['col0_Unit'] = 's'
            stps.attrs['col1'] = 'RA' if obsS.is_radec else 'Azimuth'
            stps.attrs['col1_Unit'] = 'h' if obsS.is_radec else 'd'
            stps.attrs['col2'] = 'Dec' if obsS.is_radec else 'Elevation'
            stps.attrs['col2_Unit'] = 'd'
            stps.attrs['col3'] = 'Tuning1'
            stps.attrs['col3_Unit'] = 'Hz'
            stps.attrs['col4'] = 'Tuning2'
            stps.attrs['col4_Unit'] = 'Hz'
            
            # Extract the data for the steps
            data = numpy.zeros((len(obsS.steps), 5))
            t = obsS.mjd*86400.0 + obsS.mpm/1000.0 - 3506716800.0
            for i,s in enumerate(obsS.steps):
                data[i,0] = t
                data[i,1] = s.c1
                data[i,2] = s.c2
                data[i,3] = ndp.word_to_freq(s.freq1)
                data[i,4] = ndp.word_to_freq(s.freq2)
                
                ## Update the start time for the next step
                t += s.dur / 1000.0
                
            # Save it
            stps['Steps'] = data
            
            # Deal with specified delays and gains if needed
            if obsS.steps[0].delays is not None and obsS.steps[0].gains is not None:
                nstand = 256
                label_base = 'NDP'
                if station == 'ovrolwa':
                    nstand = 352
                    label_base = 'OVR'
                    
                cbfg = grp.create_group('CustomBeamforming')
                dlys = cbfg.create_dataset('Delays', (len(obsS.steps), nstand*2+1), 'f4')
                dlys.attrs['col0'] = 'StartTime'
                dlys.attrs['col0_Unit'] = 's'
                for j in range(2*nstand):
                    dlys.attrs['col%i' % (j+1)] = '%s Digitizer %i' % (label_base, j+1)
                    dlys.attrs['col%i_Unit' % (j+1)] = 'ns'
                    
                # Extract the delays
                dataD = numpy.zeros((len(obsS.steps), nstand*2+1))
                t = obsS.mjd*86400.0 + obsS.mpm/1000.0 - 3506716800.0
                for i,s in enumerate(obsS.steps):
                    dataD[i,0] = t
                    for j in range(2*nstand):
                        try:
                            dataD[i,1+j] = _valuetoDelay(s.delays[j])
                        except IndexError as e:
                            warnings.warn("Failed to set step delay value for antenna %i: %s" % (j+1, str(e)))
                            
                # Save the delays
                dlys[:,:] = dataD
                
                gais = cbfg.create_dataset('Gains', (len(obsS.steps), nstand*2*2+1), 'f4')
                gais.attrs['col0'] = 'StartTime'
                gais.attrs['col0_Unit'] = 's'
                m = 1
                for j in range(nstand):
                    for k in range(2):
                        for l in range(2):
                            gais.attrs['col%i' % m] = '%s Stand %i %s contribution to beam %s' % (label_base, j+1, 'X' if k == 0 else 'Y', 'X' if l == 0 else 'Y')
                            gais.attrs['col%i_Unit' % m] = 'None'
                            m += 1
                            
                # Extract the gains
                dataG = numpy.zeros((len(obsS.steps), nstand*2*2+1))
                for i,s in enumerate(obsS.steps):
                    dataG[i,0] = t
                    for j in range(nstand):
                        try:
                            dataG[i,1+4*j+0] = _valuetoGain(s.gains[j][0][0])
                            dataG[i,1+4*j+1] = _valuetoGain(s.gains[j][0][1])
                            dataG[i,1+4*j+2] = _valuetoGain(s.gains[j][1][0])
                            dataG[i,1+4*j+3] = _valuetoGain(s.gains[j][1][1])
                        except IndexError as e:
                            warnings.warn("Failed to set step gains values for stand %i: %s" % (j+1, str(e)))
                            
                # Save the gains
                gais[:,:] = dataG
                
    return True


def get_observation_set(f, observation):
    """
    Return a reference to the specified observation.
    """
    
    # Get the observation
    obs = f.get('/Observation%i' % observation, None)
    if obs is None:
        raise RuntimeError('No such observation: %i' % observation)
        
    return obs


def create_observation_set(f, observation, tuning, frequency, chunks, data_products=['XX', 'YY']):
    """
    Fill in a tuning group with the right set of dummy data sets and 
    attributes.
    """
    
    # Get the observation
    obs = f.get('/Observation%i' % observation, None)
    if obs is None:
        obs = f.create_group('/Observation%i' % observation)
        
        # Target info.
        obs.attrs['TargetName'] = ''
        obs.attrs['RA'] = -99.0
        obs.attrs['RA_Units'] = 'hours'
        obs.attrs['Dec'] = -99.0
        obs.attrs['Dec_Units'] = 'degrees'
        obs.attrs['Epoch'] = 2000.0
        obs.attrs['TrackingMode'] = 'Unknown'
        
        # Observation info
        obs.attrs['ARX_Filter'] = -1.0
        obs.attrs['ARX_Gain1'] = -1.0
        obs.attrs['ARX_Gain2'] = -1.0
        obs.attrs['ARX_GainS'] = -1.0
        obs.attrs['Beam'] = -1.0
        obs.attrs['DRX_Gain'] = -1.0
        obs.attrs['sampleRate'] = -1.0
        obs.attrs['sampleRate_Units'] = 'samples/s'
        obs.attrs['tInt'] = -1.0
        obs.attrs['tInt_Units'] = 's'
        obs.attrs['LFFT'] = -1
        obs.attrs['nChan'] = -1
        obs.attrs['RBW'] = -1.0
        obs.attrs['RBW_Units'] = 'Hz'
        
    # Add the dataset that stores time as a two-element quantity
    if 'time' not in obs:
        d = obs.create_dataset('time', (chunks,), dtype=numpy.dtype({"names": ["int", "frac"],
                                                                     "formats": ["<i8", "<f8"]}))
        d.attrs['format'] = 'unix'
        d.attrs['scale'] = 'utc'
        
    # Get the group or create it if it doesn't exist
    grp = obs.get('Tuning%i' % tuning, None)
    if grp is None:
        grp = obs.create_group('Tuning%i' % tuning)
        
    grp['freq'] = frequency.astype('<f8')
    grp['freq'].attrs['Units'] = 'Hz'
    for p in data_products:
        chunk_size = HDF5_CHUNK_SIZE_MB * 1024**2 // 4 // frequency.size
        chunk_size = max([1, chunk_size])
        chunk_size = min([chunks, chunk_size])
        
        d = grp.create_dataset(p, (chunks, frequency.size), dtype='<f4', chunks=(chunk_size, frequency.size))
        d.attrs['axis0'] = 'time'
        d.attrs['axis1'] = 'frequency'
    d = grp.create_dataset('Saturation', (chunks, 2), dtype='<i8')
    d.attrs['axis0'] = 'time'
    d.attrs['axis1'] = 'polarization'
    
    return True


def get_time(f, observation):
    """
    Return a reference to the time data set for the specified observation.
    """
    
    # Get the observation
    obs = f.get('/Observation%i' % observation, None)
    if obs is None:
        raise RuntimeError('No such observation: %i' % observation)
        
    # Get the data set
    try:
        d = obs['time']
    except:
        raise RuntimeError("Unknown data set for Observation %i: %s" % (observation, 'time'))
        
    return d


def get_data_set(f, observation, tuning, dataProduct):
    """
    Return a reference to the specified data set.
    """
    
    # Catch 'time' and redirect it to get_time
    if dataProduct == 'time':
        return get_time(f, observation)
        
    # Get the observation
    obs = f.get('/Observation%i' % observation, None)
    if obs is None:
        raise RuntimeError('No such observation: %i' % observation)
        
    # Get the groups
    grp = obs.get('Tuning%i' % tuning, None)
    if grp is None:
        raise RuntimeError("Unknown tuning: %i" % tuning)
        
    # Get the data set
    try:
        d = grp[dataProduct]
    except:
        raise RuntimeError("Unknown data product for Observation %i, Tuning %i: %s" % (observation, tuning, dataProduct))
        
    return d

import atexit
atexit.register(_open_hdf_files.close_all)

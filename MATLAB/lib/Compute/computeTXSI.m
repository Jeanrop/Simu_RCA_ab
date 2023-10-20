function TXSI = computeTXSI(Trans, TX, TW)


%% -- TX SI Definition => 
for itx = numel(TX):-1:1
    itw = TX(itx).waveform;
    %-- pulseShape before the aperture transfer function (sampling at 250 Hz)
    if isfield(TW(itw),'TriLvlWvfm_Sim')
        TXSI(itx).pulseShape = TW(itw).TriLvlWvfm_Sim; % Tristate Waveform
    elseif isfield(TW(itw),'TriLvlWvfm')
        TXSI(itx).pulseShape = TW(itw).TriLvlWvfm; % Tristate Waveform
    end

    TXSI(itx).Delay         = TX(itx).Delay(:)/(Trans.frequency*1e6); % Elements Delays in Transmit (s)

    TXSI(itx).Apod          = TX(itx).Apod(:); % Elements Apodization in Transmit
end

end

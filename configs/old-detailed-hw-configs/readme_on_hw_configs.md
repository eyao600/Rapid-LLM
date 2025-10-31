The hardware configs in this directory are using 'old-style' detailed hardware models that include area/power/voltage scaling. This is still theoretically supported, but newer configs use simplified bandwidth/size/energy models that are far easier to set up and use. 

For example, in the new simpler configs, you can directly set the bandwidth and size of DRAM, and don't have to work backwards through number of stacks and other parameters. This makes hardware setup much easier.

These configs are *NOT* validated and likely inaccurate, as we internally iterated on the simplified config versions, and left these only for legacy purposes. Please double check before use!
If a detailed hardware config is desired, please copy some *sections* from these detailed configs into the simplified configs and use only the detailed *sections* you need.
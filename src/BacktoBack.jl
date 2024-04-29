using UnfoldSim 
using UnfoldMakie
using CairoMakie
using DataFrames
using Random

design = SingleSubjectDesign(conditions = Dict(:picture => ["dog","cat"])) |> x->RepeatDesign(x,10)

c = LinearModelComponent(; basis = p100(), formula = @formula(0 ~ 1+picture), β = [1,0.5]);
c2 = LinearModelComponent(; basis = p300(), formula = @formula(0 ~ 1), β = [1,-3]);

hart = headmodel(type = "hartmut")
mc = UnfoldSim.MultichannelComponent(c, hart => "Left Postcentral Gyrus")
mc2 = UnfoldSim.MultichannelComponent(c2, hart => "Right Occipital Pole")

onset = NoOnset();#UniformOnset(; width = 20, offset = 4);

data, events =
    simulate(MersenneTwister(1), design, [mc, mc2], onset, PinkNoise(noiselevel = 0.0);return_epoched=true
    )
size(data)

pos3d = hart.electrodes["pos"];
pos2d = to_positions(pos3d')
pos2d = [Point2f(p[1] + 0.5, p[2] + 0.5) for p in pos2d];

f = Figure()
df = DataFrame(
    :estimate => data[:],
    :channel => repeat(1:size(data, 1), outer = size(data, 2)),
    :time => repeat(1:size(data, 2), inner = size(data, 1)),
)
plot_butterfly!(f[1, 1:2], df; positions = pos2d)
plot_topoplot!(
    f[2, 1],
    df[df.time.==28, :];
    positions = pos2d,
    visual = (; enlarge = 0.5, label_scatter = false),
    axis = (; limits = ((0, 1), (0, 0.9))),
)
plot_topoplot!(
    f[2, 2],
    df[df.time.==48, :];
    positions = pos2d,
    visual = (; enlarge = 0.5, label_scatter = false),
    axis = (; limits = ((0, 1), (0, 0.9))),
)
f
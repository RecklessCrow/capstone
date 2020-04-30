previous_hits = 0
previous_stage = 1
center = 96
flag = true


function reward(data)
    local reward = 0

    if previous_hits < data.current_stage then
        reward = reward + 2.0
        previous_hits = previous_hits + 1
    end

    if previous_stage < data.current_stage then
        reward = reward + 5.0
        previous_stage = previous_stage + 1
    end

    local delta_x = math.abs(center - data.x_pos)
    if delta_x <= 44 then
        reward = reward + 0.0001
    else
        reward = reward - 0.0001
    end

    if is_hit != 0 and flag then
        reward = -1.0
        flag = false
    else
        flag = is_hit == 0
    end
end
function ps = mexican_hat(times, lag, scale)

    t = (times-lag)./scale;
    ps = (1- 16*t.^2).*exp(-8*t.^2);

end % funcs
function checkError(v1, v2, m1, m2, s)
if any(any(isnan(v1))), error('%s has NaNs in %s!', m1, s); end
if any(any(isnan(v2))), error('%s has NaNs in %s!', m2, s); end
if any(any(isinf(v1))), error('%s has Infs in %s!', m1, s); end
if any(any(isnan(v2))), error('%s has Infs in %s!', m2, s); end
diff = max(max(abs((v1 - v2)./v1)));
if diff > 1e-5
    error('%s and %s implementation disagree in %s.', m1, m2, s);
end
end

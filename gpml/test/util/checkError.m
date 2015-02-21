function checkError(v1, v2, m1, m2, s)
diff = max(max(abs((v1 - v2)./v1)));
if diff > 1e-5
    error('%s and %s implementation disagree in %s.', m1, m2, s);
end
end
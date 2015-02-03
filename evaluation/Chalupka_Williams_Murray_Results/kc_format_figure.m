function kc_format_figure(xlen, ylen, marginsize)
set(gcf, 'PaperUnits', 'centimeter');
set(gcf, 'PaperSize', [xlen+2*marginsize ylen+2*marginsize]);
set(gcf, 'PaperPosition', [marginsize, marginsize, xlen, ylen]);
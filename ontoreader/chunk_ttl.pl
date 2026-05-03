#!/usr/bin/perl -w --

$nspre = "";
$chdata = "";
$nchunk = 0;
$fnum = 1;

while ($r = <>) {
  $r =~ s/\s+$//g;
  if ($r =~ /^$/) {
    $nchunk += 1;
    if ($nchunk>10000) {
      open OUT, ">chunk_".$fnum.".ttl" or die "cannot write chunk_nnn.ttl";
      print STDERR "writing chunk_".$fnum.".ttl\n";
      print OUT $nspre;
      print OUT $chdata;
      close OUT or warn "problem at close";
      $chdata = "";
      $nchunk = 0;
      $fnum += 1;
    }
  }elsif ($r =~ /^\@/) {
    $nspre .= $r . "\n";
  }else {
    $chdata .= $r . "\n";
  }
}

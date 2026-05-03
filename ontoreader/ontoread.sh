#!/bin/bash
D=`dirname $0`
java -Dfile.encoding=UTF-8 -cp $D/build:\
$D/jenalib/arq-2.8.7.jar:\
$D/jenalib/icu4j-3.4.4.jar:\
$D/jenalib/iri-0.8.jar:\
$D/jenalib/jena-2.6.4.jar:\
$D/jenalib/jena-2.6.4-tests.jar:\
$D/jenalib/junit-4.5.jar:\
$D/jenalib/log4j-1.2.13.jar:\
$D/jenalib/lucene-core-2.3.1.jar:\
$D/jenalib/slf4j-api-1.5.8.jar:\
$D/jenalib/slf4j-log4j12-1.5.8.jar:\
$D/jenalib/stax-api-1.0.1.jar:\
$D/jenalib/wstx-asl-3.2.9.jar:\
$D/jenalib/xercesImpl-2.7.1.jar\
 net.pulautin.ontotool.OntoReader $*



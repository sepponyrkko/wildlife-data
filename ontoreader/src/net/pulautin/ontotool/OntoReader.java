package net.pulautin.ontotool;

import java.io.File;

import com.hp.hpl.jena.ontology.OntModel;
import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.ModelFactory;
import com.hp.hpl.jena.rdf.model.Property;
import com.hp.hpl.jena.rdf.model.RDFNode;
import com.hp.hpl.jena.rdf.model.Resource;
import com.hp.hpl.jena.rdf.model.Statement;
import com.hp.hpl.jena.rdf.model.StmtIterator;

public class OntoReader {

	static OntModel omm = ModelFactory.createOntologyModel();
	
	public OntoReader() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		if(args.length != 2) {
			System.err.println("usage: ontoread file lang");
			return;
		}
		
		omm.setNsPrefix("mesh", "http://purl.bioontology.org/ontology/MESH/");
		omm.setNsPrefix("sty", "http://purl.bioontology.org/ontology/STY/");
		omm.setNsPrefix("n", "file:/nature/");
//		omm.setNsPrefix("http://purl.bioontology.org/ontology/MESH", "mesh:");

		Model om = ModelFactory.createDefaultModel();
		String lang = args[1];
		String path = args[0];
		om.read((new File(path)).toURI().toString(),lang);
		
		StmtIterator si = om.listStatements();
		while(si.hasNext()){
			Statement s = si.nextStatement();
			Resource r = s.getSubject();
			Property p = s.getPredicate();
			RDFNode o = s.getObject();
			
			String row = nodeToString(r)+"\t"+
				nodeToString(p)+"\t"+
				nodeToString(o);			
			
			
			System.out.println(row);
		}
		

	}

	private static String nodeToString(RDFNode o) {
		String s = nodeToStringX(o);
		s = s.replaceAll("\\s+"," ");
		return s;
	}
	private static String nodeToStringX(RDFNode o) {
		
		if (o.isLiteral()) 
			return o.toString().replaceAll("\\\"", "").replaceAll("\\^\\^.*", "");
	
		
		if (o.isAnon()) 
			return o.toString();
		
		
		if (o.canAs(Resource.class)) {
			String s = o.as(Resource.class).toString();			
			return omm.shortForm(o.getModel().shortForm(s)); 
		}
		
		return "unknown";
	}

}

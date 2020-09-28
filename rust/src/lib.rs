extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
#[macro_use]
extern crate quick_error;
extern crate petgraph;
extern crate vosealias;
extern crate rand;

use std::collections::{HashMap, HashSet};
use std::slice;
use std::io::Write;
use petgraph::graph::{NodeIndex};
use vosealias::AliasTable;

#[derive(Deserialize)]
struct NodeLinkGraph {
    directed: bool,
    #[allow(dead_code)]
    multigraph: bool,
    #[allow(dead_code)]
    graph: GraphInfo,
    links: Vec<Link>,
    nodes: Vec<Node>,
}
#[derive(Deserialize)]
struct GraphInfo {
    #[allow(dead_code)]
    name: Option<String>
}
#[derive(Deserialize)]
struct Link {
    source: i64,
    target: i64,
}
#[derive(Deserialize)]
struct Node {
    id: i64,
}

#[repr(C)]
pub enum ErrorCode {
    NoError = 0,
    InvalidGraph,
}

quick_error! {
    #[derive(Debug)]
    pub enum InternalError {
        Deserializing(err: serde_json::Error) {
            from()
        }
    }
}

pub struct Answer {
    vec: Vec<u8>
}

#[no_mangle]
pub extern fn destroy(ptr: *mut Answer) {
    unsafe {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
pub extern fn get_buffer(ptr: *mut Answer) -> *const u8 {
    unsafe {
        (*ptr).vec.as_ptr()
    }
}
#[no_mangle]
pub extern fn get_length(ptr: *mut Answer) -> usize {
    unsafe {
        (*ptr).vec.len()
    }
}

#[no_mangle]
pub extern fn precompute_probabilities(ptr: *const u8,
                                       length: usize,
                                       p: f64, q: f64,
                                       num_walks: usize,
                                       walk_length: usize) -> *mut Answer {
    let slice = unsafe { slice::from_raw_parts(ptr, length) };

    match safe_precompute_probabilities(slice, p, q, num_walks, walk_length) {
        Ok(result) => {
            Box::into_raw(Box::new(Answer { vec: result }))
        }
        Err(InternalError::Deserializing(e)) => {
            eprintln!("failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

type Probs<K> = HashMap<K, AliasTable<NodeIndex, f64>>;

fn generate_walks<Ty: petgraph::EdgeType>
    (graph: &G<Ty>,
     num_walks: usize,
     length: usize,
     first_probs: &Probs<NodeIndex>,
     second_probs: &Probs<(NodeIndex, NodeIndex)>)
    -> Vec<u8>
{
    let mut rng = rand::weak_rng();

    let mut walks = Vec::new();
    for n in graph.node_indices() {
        for _ in 0..num_walks {
            let mut two_ago = n;
            let mut one_ago = n;
            if !walks.is_empty() {
                walks.push(b'\n')
            }
            write!(&mut walks, "{}", graph.node_weight(n).unwrap());

            for len in 1..length {
                let p = if len == 1 {
                    first_probs.get(&one_ago)
                } else {
                    second_probs.get(&(two_ago, one_ago))
                };
                match p {
                    Some(p) => {
                        let next = *p.pick(&mut rng);
                        two_ago = one_ago;
                        one_ago = next;
                        write!(&mut walks, " {}", graph.node_weight(next).unwrap());
                    }
                    None => break
                }
            }
        }
    }

    walks
}

type G<Ty> = petgraph::Graph<i64, (), Ty>;
fn do_it<Ty: petgraph::EdgeType>(
    nlg: &NodeLinkGraph, p: f64, q: f64, num_walks: usize, walk_length: usize
) -> Vec<u8> {
    assert!(nlg.directed == Ty::is_directed());
    let mut map = HashMap::new();

    let mut graph = G::<Ty>::with_capacity(nlg.nodes.len(), nlg.links.len());

    for n in &nlg.nodes {
        let id = graph.add_node(n.id);
        map.insert(n.id, id);
    }

    for link in &nlg.links {
        graph.add_edge(map[&link.source],
                       map[&link.target],
                       ());
    }

    let mut first_probs = HashMap::new();
    let mut second_probs = HashMap::new();
    for source in graph.node_indices() {
        let source_neighbors = graph.neighbors(source).collect::<Vec<_>>();
        let source_neighbors_set = source_neighbors.iter().cloned().collect::<HashSet<_>>();

        let mut first_travel_weights = Vec::with_capacity(source_neighbors.len());
        for &current_node in &source_neighbors {
            let current_weight = 1_f64;
            first_travel_weights.push((current_node, current_weight));

            let mut unnormalized =
                graph.neighbors(current_node)
                .map(|destination| {
                    let raw_weight = 1_f64;
                    let weight = if destination == source {
                        raw_weight / p
                    } else if source_neighbors_set.contains(&destination) {
                        raw_weight
                    } else {
                        raw_weight / q
                    };
                    (destination, weight)
                })
                .peekable();


            if unnormalized.peek().is_some() {
                let alias = unnormalized.collect::<AliasTable<_, _>>();
                second_probs.insert((source, current_node), alias);
            }
        }

        if !first_travel_weights.is_empty() {
            let alias = first_travel_weights.into_iter().collect::<AliasTable<_, _>>();
            first_probs.insert(source, alias);
        }
    }
    println!("generating walks");
    let walks = generate_walks(&graph, num_walks, walk_length,
                               &first_probs, &second_probs);

    walks
}

fn safe_precompute_probabilities(json: &[u8], p: f64, q: f64,
                                 num_walks: usize, length: usize) -> Result<Vec<u8>, InternalError> {
    let loaded: NodeLinkGraph = serde_json::from_slice(json)?;

    let result = if loaded.directed {
        do_it::<petgraph::Directed>(&loaded, p, q, num_walks, length)
    } else {
        do_it::<petgraph::Undirected>(&loaded, p, q, num_walks, length)
    };
    Ok(result)
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

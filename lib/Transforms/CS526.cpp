#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h" 
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>

namespace mlir {
#define GEN_PASS_DEF_CS526
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct CS526 : public impl::CS526Base<CS526> {

struct OpDetail {
    std::vector<std::string> consumers;
    std::vector<std::string> inputTensorDetails; // To store tensor type and size
};

std::unordered_map<std::string, OpDetail> opConsumers;
std::unordered_map<std::string, OpDetail> dataFlowGraph;
std::set<std::string> visited;
std::vector<std::string> order;



  void runOnOperation() override {
    func::FuncOp function = getOperation();
    function.emitRemark("Running CS526Pass on function: ") << function.getName();
    populateConsumerMap(function);
    //printConsumerMap();

    // Build and print the data flow graph
    buildDataFlowGraph();
    //printDataFlowGraph();  // Optional: Console output of the data flow graph
    removeUnwantedEdges();
    removeAndRedirectClampNodes();
    //printDataFlowGraph();
    //printAllKeys();

        // Use user input to determine the starting node 
    std::string startNode = findStartingNode("15:11");

    if (!startNode.empty()) {
        llvm::errs() << "Starting DFS from node: " << startNode << "\n";
        dfs(startNode, visited, order);
        printOrder(order);
    } else {
        llvm::errs() << "No valid starting node provided or found.\n";
    }



    generateProgramStringsToFile();

    //llvm::errs()<< "Check parameter size of " << "loc(\"real_lenet.mlir\":24:11) tosa.conv2d" << checkParameterSize("loc(\"real_lenet.mlir\":24:11) tosa.conv2d") << "\n";

    //llvm::errs()<< "Check parameter size of " << "loc(\"real_lenet.mlir\":33:11) tosa.conv2d" << checkParameterSize("loc(\"real_lenet.mlir\":33:11) tosa.conv2d") << "\n";
    // Generate a DOT file for visualization
    generateDotFile("dataflow.dot");




    // function.walk([](Operation *op) {

    //   // TOSA conv2d case: 
    //   if (auto convOp = dyn_cast<mlir::tosa::Conv2DOp>(op)) {
    //     llvm::outs() << "conv detected\n";  // Print "conv" if it's a tosa.conv2d operation
    //     convOp->print(llvm::outs());
    //     llvm::outs() << "\n";
    //   }

    //   //TOSA FC case: 
    //   else if (auto fcOp = dyn_cast<mlir::tosa::FullyConnectedOp>(op)) {
    //     llvm::outs() << "fully_connected\n";  // Print "fully_connected" if it's a tosa.fully_connected operation
    //     fcOp->print(llvm::outs());
    //     llvm::outs() << "\n"; 
    //   }

    //   //TOSA FC case: 
    //   else if (auto maxpoolOp = dyn_cast<mlir::tosa::MaxPool2dOp>(op)) {
    //     llvm::outs() << "maxpool_2d\n";  // Print "fully_connected" if it's a tosa.fully_connected operation
    //     maxpoolOp->print(llvm::outs());
    //     llvm::outs() << "\n"; 
    //   }

    // });
  }




std::string getOperationId(mlir::Operation *op) {
    if (!op->getResults().empty()) {
        // Get the first result and its defining operation
        mlir::Operation* defOp = op->getResult(0).getDefiningOp();
        std::string resultName = defOp->getName().getStringRef().str();
        std::string locStr;
        llvm::raw_string_ostream rso(locStr);
        defOp->getLoc().print(rso);
        rso.flush();

        // Attempt to extract just the identifier part, assuming it starts with '%'
        size_t idx = locStr.find('%');
        if (idx != std::string::npos) {
            size_t end = locStr.find(' ', idx); // Find the space after the identifier
            if (end != std::string::npos)
                locStr = locStr.substr(idx, end - idx); // Get substring from '%' to the first space
            else
                locStr = locStr.substr(idx); // No space found, take the rest of the string
        }
        return locStr + " " + resultName;
    } else {
        // Fallback to using a combination of operation type and address
        return op->getName().getStringRef().str() + "@" + std::to_string(reinterpret_cast<uintptr_t>(op));
    }
}


//TODO: Cannot handle argument correctly still 

void populateConsumerMap(func::FuncOp function) {
    // Initialize consumers for arguments
    for (auto arg : function.getArguments()) {
        std::string argId = "arg" + std::to_string(arg.getArgNumber());
        opConsumers[argId] = OpDetail{};  // Initialize with empty details
    }

    // Walk through the function to populate consumer map
    function.walk([this](Operation *op) {
       // llvm::errs() << "Starting to process operation: " << getOperationId(op) << "\n";

        std::vector<std::string> inputTensorDetails;
        for (const auto &operand : op->getOperands()) {
            // Gather tensor type information
            inputTensorDetails.push_back(formatType(operand.getType()));

            if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
                std::string argId = "arg" + std::to_string(blockArg.getArgNumber());
                addConsumer(argId, getOperationId(op)); // Directly link argument to operation
                continue;
            }
            

            Operation *currentOp = operand.getDefiningOp();
            if (!currentOp) {
               // llvm::errs() << "Found an operand without a defining operation, possibly an external input.\n";
                continue;
            }

            currentOp = traceToSource(currentOp);
            if (currentOp) {
                addConsumer(getOperationId(currentOp), getOperationId(op));
            } else {
                //llvm::errs() << "Reached a null operation in processing.\n";
            }
        }

        opConsumers[getOperationId(op)].inputTensorDetails = inputTensorDetails;
    });

}



// void populateConsumerMap(func::FuncOp function) {
//     // Initialize consumers for arguments
//     for (auto arg : function.getArguments()) {
//         std::string argId = "arg" + std::to_string(arg.getArgNumber());
//         opConsumers[argId] = OpDetail{};  // Initialize with empty details
//     }

//     // Walk through the function to populate consumer map
//     function.walk([this](Operation *op) {
//         llvm::errs() << "Starting to process operation: " << getOperationId(op) << "\n";

//         std::vector<std::string> inputTensorDetails;
//         for (const auto &operand : op->getOperands()) {
//             // Gather tensor type information
//             inputTensorDetails.push_back(formatType(operand.getType()));

//             // Check if operand is a block argument (function argument)
//             if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
//                 std::string argId = "arg" + std::to_string(blockArg.getArgNumber());
//                 // Here we should ensure to only trace from the current operation, not the block argument
//                 Operation *significantOp = traceToSource(op); // Start tracing from the operation that uses the argument
//                 if (significantOp != op) { // Ensure we find a different operation than the current one
//                     addConsumer(argId, getOperationId(significantOp)); // Link argument to the significant operation found
//                 } else {
//                     // In case no other operations are found, link directly to the current operation
//                     addConsumer(argId, getOperationId(op));
//                 }
//             }
//         }

//         opConsumers[getOperationId(op)].inputTensorDetails = inputTensorDetails;
//     });
// }






void addConsumer(const std::string &producerId, const std::string &consumerId) {
    auto &consumers = opConsumers[producerId].consumers;
    if (std::find(consumers.begin(), consumers.end(), consumerId) == consumers.end()) {
        consumers.push_back(consumerId);
    }
}



Operation* traceToSource(Operation* currentOp) {
    while (currentOp && !currentOp->getOperands().empty() &&
           (currentOp->getName().getStringRef() == "tosa.reshape" ||
            currentOp->getName().getStringRef() == "tosa.transpose")) {
        //llvm::errs() << "Skipping intermediate operation: " << getOperationId(currentOp) << "\n";
        currentOp = currentOp->getOperands().front().getDefiningOp();
    }
    return currentOp;
}




void printConsumerMap() {
    llvm::errs() << "Printing Consumer Map:\n";
    for (const auto &entry : opConsumers) {
        llvm::errs() << "Producer [" << entry.first << "] has consumers:\n";
        for (const auto &consumer : entry.second.consumers) {
            llvm::errs() << "  - " << consumer << "\n";
        }
        llvm::errs() << "Input Tensors: ";
        for (const auto &tensorDetail : entry.second.inputTensorDetails) {
            llvm::errs() << tensorDetail << "; ";
        }
        llvm::errs() << "\n";
    }
}



void buildDataFlowGraph() {
    for (auto& entry : opConsumers) {
        std::string producer = entry.first;
        const auto &details = entry.second;

        // Check if the producer is an argument
        bool isArgument = producer.rfind("arg", 0) == 0;

        if (isArgument) {
            // Handle argument case separately and print
           // std::cout << "Argument: " << producer << std::endl;
            if (!details.consumers.empty()) {
               // std::cout << "  Consumers: ";
                for (const auto& consumer : details.consumers) {
                    std::cout << consumer << ", ";
                }
                std::cout << std::endl;
            }
        } else {
            // Filter out producers that only lead to reshape or transpose operations
            if (!std::all_of(details.consumers.begin(), details.consumers.end(), [this](const std::string &consumerId) {
                return isReshapeOrTranspose(consumerId);
            })) {
                // For non-argument producers, add consumers to the dataflow graph and print
                for (const auto& consumer : details.consumers) {
                    dataFlowGraph[producer].consumers.push_back(consumer);
                    dataFlowGraph[producer].inputTensorDetails = details.inputTensorDetails;
                }
            }
        }
    }
}


void removeUnwantedEdges() {
    // Iterate over each node in the graph
    for (auto &node : dataFlowGraph) {
        std::vector<std::string> &consumers = node.second.consumers;
        
        // Use the erase-remove idiom to remove unwanted edges
        consumers.erase(
            std::remove_if(consumers.begin(), consumers.end(),
                [this](const std::string &consumer) {
                    return isReshapeOrTranspose(consumer);
                }),
            consumers.end()
        );
    }
}



bool isReshapeOrTranspose(const std::string &operationId) {
    // Check if the operation ID includes substring indicating a reshape or transpose
    return operationId.find("tosa.reshape") != std::string::npos ||
           operationId.find("tosa.transpose") != std::string::npos;
}

void removeAndRedirectClampNodes() {
    std::unordered_map<std::string, std::vector<std::string>> redirects;

    // First pass: Collect redirections and prepare for removal of clamp nodes
    for (const auto &entry : dataFlowGraph) {
        const std::string &producer = entry.first;
        if (isClamp(producer)) {
            // For each consumer of clamp, all of producer's consumers need to be redirected
            for (const std::string &prodConsumer : entry.second.consumers) {
                redirects[producer].push_back(prodConsumer);
            }
        }
    }

    // Second pass: Redirect connections and remove clamp nodes
    std::unordered_map<std::string, OpDetail> newGraph;
    for (const auto &entry : dataFlowGraph) {
        const std::string &producer = entry.first;

        if (!isClamp(producer)) { // Retain non-clamp nodes
            OpDetail &detail = newGraph[producer];
            detail.inputTensorDetails = entry.second.inputTensorDetails;

            // Check each consumer if it is a clamp and needs redirection
            for (const std::string &consumer : entry.second.consumers) {
                if (redirects.find(consumer) != redirects.end()) {
                    // Redirect to the clamp's consumers
                    detail.consumers.insert(detail.consumers.end(), redirects[consumer].begin(), redirects[consumer].end());
                } else {
                    detail.consumers.push_back(consumer);
                }
            }
        }
    }

    // Replace old graph with the new graph
    dataFlowGraph = newGraph;
}

bool isClamp(const std::string &operationId) {
    // Check if the operation is a clamp
    return operationId.find("clamp") != std::string::npos;
}



void printDataFlowGraph() {
    llvm::errs() << "Data Flow Graph:\n";
    for (auto& node : dataFlowGraph) {
        llvm::errs() << "Operation: " << node.first << " flows to:\n";
        for (auto& target : node.second.consumers) { // Correctly accessing the consumers vector
            llvm::errs() << "  -> " << target << "\n";
        }
        llvm::errs() << "Input Tensors: ";
        for (auto& tensor : node.second.inputTensorDetails) { // Correctly accessing the input tensor details
            llvm::errs() << tensor << "; ";
        }
        llvm::errs() << "\n";
    }
}




void generateDotFile(const std::string& filename) {
    std::ofstream dotFile(filename);
    if (!dotFile.is_open()) {
        llvm::errs() << "Failed to open file: " << filename << "\n";
        return;
    }

    dotFile << "digraph DataFlow {\n";
    std::set<std::pair<std::string, std::string>> edges;  // To track and prevent duplicate edges

    for (const auto& node : dataFlowGraph) {
        std::string formattedProducer = formatDotIdentifier(node.first);
        for (const auto& consumer : node.second.consumers) {
            std::string formattedConsumer = formatDotIdentifier(consumer);
            // Insert the edge into the set and check if it was already there
            if (edges.insert({formattedProducer, formattedConsumer}).second) {
                dotFile << "\"" << formattedProducer << "\" -> \"" << formattedConsumer << "\";\n";
            }
        }
    }
    dotFile << "}\n";
    dotFile.close();
}




std::string formatDotIdentifier(const std::string& identifier) {
    std::string result;
    for (char c : identifier) {
        switch (c) {
            case '"': 
            case '\\':
                result += '\\'; // Escape backslashes and double quotes
                result += c;
                break;
            case '(':
            case ')':
            case ':':
            case ' ':
            case '.':
                result += '_'; // Replace other problematic characters with underscore
                break;
            default:
                result += c;
        }
    }
    return result;
}

std::string formatType(mlir::Type type) {
    std::string result;
    llvm::raw_string_ostream rso(result);
    if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
        rso << tensorType.getElementType() << "[";
        for (int i = 0; i < tensorType.getRank(); ++i) {
            rso << tensorType.getDimSize(i);
            if (i < tensorType.getRank() - 1) rso << "x";
        }
        rso << "]";
    } else {
        type.print(rso);
    }
    return result;
}


std::string findStartingNode(const std::string& substring) {
    llvm::errs() << "Searching for node containing: '" << substring << "'\n";
    std::string candidateNode;

    // Search for any node containing the substring
    for (const auto& entry : dataFlowGraph) {
        if (entry.first.find(substring) != std::string::npos) {
            candidateNode = entry.first;  // Consider this node as a candidate
            break;  // Stop at the first match
        }
    }

    if (candidateNode.empty()) {
        llvm::errs() << "No node containing substring found.\n";
        return "";
    }


    // If the candidate node has no incoming edges, it's a valid start node
    llvm::errs() << "Valid starting node found: '" << candidateNode << "'\n";
    return candidateNode;
}



void dfs(const std::string& node, std::set<std::string>& visited, std::vector<std::string>& order) {
    if (visited.find(node) != visited.end()) return;
    visited.insert(node);
    order.push_back(node); // This vector order could be used to track the node processing order.

    for (const auto& consumer : dataFlowGraph[node].consumers) {
        dfs(consumer, visited, order);
    }
}

void printOrder(const std::vector<std::string>& order) {
    llvm::errs() << "Execution Order:\n";
    for (const auto& node : order) {
        llvm::errs() << node << "\n";
    }
}

void printAllKeys() {
    llvm::errs() << "Printing all keys in the Data Flow Graph:\n";
    for (const auto& entry : dataFlowGraph) {
        llvm::errs() << "Key: '" << entry.first << "'\n";
    }
}


bool checkParameterSize(const std::string& operationKey) {
    // Define the threshold for the total number of elements in the tensor
    const int threshold = 100 * 5 * 5 * 16; // 40,000

    // Check if the operation exists in the dataFlowGraph
    auto it = dataFlowGraph.find(operationKey);
    if (it == dataFlowGraph.end()) {
        llvm::errs() << "Operation key does not exist in the dataFlowGraph.\n";
        return false; // or handle error differently
    }

    const OpDetail& detail = it->second;
    
    // Iterate over each tensor description in the inputTensorDetails
    for (const std::string& tensorDesc : detail.inputTensorDetails) {
        llvm::errs() << "Checking tensor sizes in: " << tensorDesc << "\n";

        // Example tensor description: "f32[1x10x60]"
        // Extract the dimensions "1x10x60"
        size_t startPos = tensorDesc.find('[');
        size_t endPos = tensorDesc.find(']');
        if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
            std::string dimensions = tensorDesc.substr(startPos + 1, endPos - startPos - 1);
            std::istringstream dimStream(dimensions);
            std::string dim;
            int totalElements = 1;
            
            // Split dimensions by 'x' and compute the product of all dimensions
            while (std::getline(dimStream, dim, 'x')) {
                totalElements *= std::stoi(dim);
            }

            // Check if the total number of elements exceeds the threshold
            if (totalElements > threshold) {
                llvm::errs() << "Tensor total elements " << totalElements << " exceed threshold " << threshold << ".\n";
                return true; // Found a tensor with elements exceeding the threshold
            }
        }
    }

    return false; // No tensor exceeded the threshold
}


void generateProgramStringsToFile() {
    std::map<std::string, std::string> operationNames = {
        {"tosa.conv2d", "convR8_32_5"},
        {"tosa.max_pool2d", "maxp2_2"},
        {"tosa.matmul", "fc128_64"}
    };

    std::ofstream outFile("output_programs.txt");
    if (!outFile) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }

    int programCounter = 1;

    bool skipNext = false;  // New flag to control skipping next operation if needed

    for (auto& entry : order) {
        std::string& operationKey = entry;
        OpDetail& detail = dataFlowGraph[entry];

        if (skipNext) {
            skipNext = false;
            continue;  // Skip this operation because the previous one handled it
        }

        std::string opType;
        for (const auto& pair : operationNames) {
            if (operationKey.find(pair.first) != std::string::npos) {
                opType = pair.first;
                break;
            }
        }

        if (opType.empty()) {
            continue;  // Skip operations not defined in operationNames
        }

        bool isLargeTensor = checkParameterSize(operationKey);
        outFile << "program " << programCounter++ << " " << operationNames[opType] << std::endl;

        if (isLargeTensor) {
            outFile << "program " << programCounter++ << " " << operationNames[opType] << std::endl;
        }

        if (opType == "tosa.matmul" && !detail.consumers.empty() && detail.consumers[0].find("tosa.add") != std::string::npos) {
            skipNext = true;  // Set flag to skip the next operation ("tosa.add")
        }
    }

    //TODO: Generate the memory interface 
    std::vector<std::string> staticStrings = {
        "memcpy2device image 0 1024",
        "memcpy2device conv1_weights 4096 150",
        "memcpy2device conv1_bias 4696 6",
        "memcpy2device conv3_weights 28240 2400",
        "memcpy2device conv3_bias 37840 16",
        "memcpy2device conv5_weights 45904 48000",
        "memcpy2device conv5_bias 237904 120",
        "memcpy2device fc6_weights 238864 1200",
        "memcpy2device fc6_bias 243664 10",
        "convR8_32_5 1 2 MS 0 4096 4696 -1 1 32 32 6",
        "maxp2_2 2 3 SS -1 -1 6 28 28 2 2",
        "convR8_32_5 3 4 SS -1 28240 37840 -1 6 14 14 16",
        "maxp2_2 4 5-6 SS -1 -1 16 10 10 2 2",
        "convR8_32_5 5 7 SS -1 45904 237904 -1 16 5 5 60",
        "convR8_32_5 6 7 SS -1 141904 238144 -1 16 5 5 60",
        "fc128_64 7 7 SM -1 238864 243664 243704 120 10",
        "memcpy2host output 243704 10"
    };

    for (const std::string& line : staticStrings) {
        outFile << line << std::endl;
    }




    outFile.close();
}






};
} // namespace

std::unique_ptr<Pass> mlir::createCS526Pass() { return std::make_unique<CS526>(); }

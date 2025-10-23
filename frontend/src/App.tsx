import React, { useState, useEffect } from 'react';
import {
  ChakraProvider,
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  Input,
  Textarea,
  Select,
  Progress,
  Alert,
  AlertIcon,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Card,
  CardBody,
  CardHeader,
  Badge,
  Spinner,
  useToast,
  Grid,
  GridItem,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Image,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  IconButton,
  Tooltip,
} from '@chakra-ui/react';
import {
  FiUpload,
  FiPlay,
  FiPause,
  FiDownload,
  FiSettings,
  FiEye,
  FiImage,
  FiType,
  FiVideo,
  FiMic,
  FiCpu,
  FiActivity,
  FiBarChart3,
  FiUsers,
  FiClock,
  FiCheckCircle,
  FiXCircle,
  FiRefreshCw,
} from 'react-icons/fi';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  ArcElement,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend,
  ArcElement
);

// Types
interface AnalysisResult {
  id: string;
  type: 'image' | 'text' | 'video' | 'audio';
  status: 'processing' | 'completed' | 'failed';
  result?: any;
  error?: string;
  timestamp: string;
  processingTime?: number;
}

interface ModelInfo {
  name: string;
  type: string;
  status: 'loaded' | 'loading' | 'unloaded';
  memoryUsage: number;
  inferenceSpeed: string;
}

interface SystemStats {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  activeUsers: number;
  modelsLoaded: number;
  gpuUsage: number;
  memoryUsage: number;
}

const App: React.FC = () => {
  // State
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [textInput, setTextInput] = useState('');
  const [analysisType, setAnalysisType] = useState('comprehensive');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  
  // Hooks
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Effects
  useEffect(() => {
    fetchSystemStats();
    fetchModels();
    const interval = setInterval(() => {
      fetchSystemStats();
    }, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  // API Functions
  const fetchSystemStats = async () => {
    try {
      const response = await fetch('/api/v1/health');
      const data = await response.json();
      
      // Mock system stats for demo
      setSystemStats({
        totalRequests: 15420,
        successfulRequests: 14891,
        failedRequests: 529,
        averageResponseTime: 1.2,
        activeUsers: 47,
        modelsLoaded: 8,
        gpuUsage: 67,
        memoryUsage: 78,
      });
    } catch (error) {
      console.error('Failed to fetch system stats:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/v1/models');
      const data = await response.json();
      
      // Mock models data
      setModels([
        { name: 'CLIP-ViT-Large', type: 'Vision', status: 'loaded', memoryUsage: 2.1, inferenceSpeed: 'Fast' },
        { name: 'BLIP-Large', type: 'Vision', status: 'loaded', memoryUsage: 1.8, inferenceSpeed: 'Medium' },
        { name: 'GPT-2-Medium', type: 'Language', status: 'loaded', memoryUsage: 1.2, inferenceSpeed: 'Fast' },
        { name: 'Stable Diffusion', type: 'Generation', status: 'loaded', memoryUsage: 3.4, inferenceSpeed: 'Slow' },
        { name: 'T5-Base', type: 'Language', status: 'unloaded', memoryUsage: 0, inferenceSpeed: 'Medium' },
      ]);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const analyzeContent = async () => {
    if (!selectedFile && !textInput.trim()) {
      toast({
        title: 'No content provided',
        description: 'Please upload a file or enter text to analyze.',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsProcessing(true);
    const startTime = Date.now();

    try {
      let result;
      const analysisId = `analysis_${Date.now()}`;

      if (selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('analysis_type', analysisType);

        const endpoint = selectedFile.type.startsWith('image/') 
          ? '/api/v1/analyze/image'
          : '/api/v1/stream/video';

        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData,
        });

        result = await response.json();
      } else {
        const response = await fetch('/api/v1/analyze/text', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: textInput,
            analysis_type: analysisType,
          }),
        });

        result = await response.json();
      }

      const processingTime = Date.now() - startTime;

      const analysisResult: AnalysisResult = {
        id: analysisId,
        type: selectedFile 
          ? (selectedFile.type.startsWith('image/') ? 'image' : 'video')
          : 'text',
        status: result.success ? 'completed' : 'failed',
        result: result.success ? result : undefined,
        error: result.success ? undefined : result.error,
        timestamp: new Date().toISOString(),
        processingTime,
      };

      setAnalysisResults(prev => [analysisResult, ...prev]);

      toast({
        title: result.success ? 'Analysis completed' : 'Analysis failed',
        description: result.success 
          ? `Processed in ${processingTime}ms`
          : result.error,
        status: result.success ? 'success' : 'error',
        duration: 5000,
        isClosable: true,
      });

    } catch (error) {
      const analysisResult: AnalysisResult = {
        id: `analysis_${Date.now()}`,
        type: selectedFile ? 'image' : 'text',
        status: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      };

      setAnalysisResults(prev => [analysisResult, ...prev]);

      toast({
        title: 'Analysis failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const generateContent = async (prompt: string, contentType: string) => {
    try {
      const response = await fetch('/api/v1/generate/content', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          content_type: contentType,
          style: 'creative',
        }),
      });

      const result = await response.json();

      if (result.success) {
        toast({
          title: 'Content generated successfully',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        return result.content;
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      toast({
        title: 'Generation failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Chart data
  const performanceChartData = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    datasets: [
      {
        label: 'Requests/Hour',
        data: [120, 89, 156, 234, 189, 167],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.4,
      },
      {
        label: 'Response Time (ms)',
        data: [800, 950, 1200, 1100, 980, 1050],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.4,
        yAxisID: 'y1',
      },
    ],
  };

  const modelUsageData = {
    labels: ['CLIP', 'BLIP', 'GPT-2', 'Stable Diffusion', 'T5'],
    datasets: [
      {
        data: [35, 25, 20, 15, 5],
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
        ],
      },
    ],
  };

  return (
    <ChakraProvider>
      <Box minH="100vh" bg="gray.50">
        {/* Header */}
        <Box bg="white" shadow="sm" px={6} py={4}>
          <HStack justify="space-between">
            <HStack>
              <FiCpu size={24} color="blue.500" />
              <Heading size="lg" color="gray.800">
                Multi-Modal AI Analyzer
              </Heading>
            </HStack>
            <HStack>
              <Badge colorScheme="green" variant="subtle">
                {systemStats?.modelsLoaded || 0} Models Loaded
              </Badge>
              <Badge colorScheme="blue" variant="subtle">
                {systemStats?.activeUsers || 0} Active Users
              </Badge>
              <IconButton
                aria-label="Refresh"
                icon={<FiRefreshCw />}
                onClick={fetchSystemStats}
                variant="ghost"
              />
            </HStack>
          </HStack>
        </Box>

        {/* Main Content */}
        <Box p={6}>
          <Tabs index={activeTab} onChange={setActiveTab}>
            <TabList>
              <Tab>
                <HStack>
                  <FiEye />
                  <Text>Analysis</Text>
                </HStack>
              </Tab>
              <Tab>
                <HStack>
                  <FiImage />
                  <Text>Generation</Text>
                </HStack>
              </Tab>
              <Tab>
                <HStack>
                  <FiBarChart3 />
                  <Text>Analytics</Text>
                </HStack>
              </Tab>
              <Tab>
                <HStack>
                  <FiSettings />
                  <Text>Models</Text>
                </HStack>
              </Tab>
            </TabList>

            <TabPanels>
              {/* Analysis Tab */}
              <TabPanel>
                <Grid templateColumns="1fr 1fr" gap={6}>
                  <GridItem>
                    <Card>
                      <CardHeader>
                        <Heading size="md">Content Analysis</Heading>
                      </CardHeader>
                      <CardBody>
                        <VStack spacing={4} align="stretch">
                          {/* File Upload */}
                          <Box>
                            <Text mb={2} fontWeight="medium">Upload File</Text>
                            <Input
                              type="file"
                              accept="image/*,video/*,audio/*"
                              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                              p={1}
                            />
                          </Box>

                          {/* Text Input */}
                          <Box>
                            <Text mb={2} fontWeight="medium">Or Enter Text</Text>
                            <Textarea
                              placeholder="Enter text to analyze..."
                              value={textInput}
                              onChange={(e) => setTextInput(e.target.value)}
                              rows={4}
                            />
                          </Box>

                          {/* Analysis Type */}
                          <Box>
                            <Text mb={2} fontWeight="medium">Analysis Type</Text>
                            <Select
                              value={analysisType}
                              onChange={(e) => setAnalysisType(e.target.value)}
                            >
                              <option value="quick">Quick Analysis</option>
                              <option value="comprehensive">Comprehensive</option>
                              <option value="detailed">Detailed</option>
                              <option value="creative">Creative</option>
                            </Select>
                          </Box>

                          {/* Analyze Button */}
                          <Button
                            colorScheme="blue"
                            onClick={analyzeContent}
                            isLoading={isProcessing}
                            loadingText="Analyzing..."
                            leftIcon={<FiPlay />}
                            size="lg"
                          >
                            Analyze Content
                          </Button>
                        </VStack>
                      </CardBody>
                    </Card>
                  </GridItem>

                  <GridItem>
                    <Card>
                      <CardHeader>
                        <Heading size="md">Analysis Results</Heading>
                      </CardHeader>
                      <CardBody>
                        <VStack spacing={3} align="stretch" maxH="500px" overflowY="auto">
                          {analysisResults.length === 0 ? (
                            <Text color="gray.500" textAlign="center">
                              No analysis results yet
                            </Text>
                          ) : (
                            analysisResults.map((result) => (
                              <Card key={result.id} size="sm" variant="outline">
                                <CardBody>
                                  <HStack justify="space-between" mb={2}>
                                    <HStack>
                                      {result.type === 'image' && <FiImage />}
                                      {result.type === 'text' && <FiType />}
                                      {result.type === 'video' && <FiVideo />}
                                      {result.type === 'audio' && <FiMic />}
                                      <Text fontWeight="medium" fontSize="sm">
                                        {result.type.toUpperCase()}
                                      </Text>
                                    </HStack>
                                    <HStack>
                                      {result.status === 'completed' && (
                                        <FiCheckCircle color="green" />
                                      )}
                                      {result.status === 'failed' && (
                                        <FiXCircle color="red" />
                                      )}
                                      {result.status === 'processing' && (
                                        <Spinner size="sm" />
                                      )}
                                      <Badge
                                        colorScheme={
                                          result.status === 'completed'
                                            ? 'green'
                                            : result.status === 'failed'
                                            ? 'red'
                                            : 'yellow'
                                        }
                                        size="sm"
                                      >
                                        {result.status}
                                      </Badge>
                                    </HStack>
                                  </HStack>
                                  
                                  {result.processingTime && (
                                    <Text fontSize="xs" color="gray.600">
                                      Processed in {result.processingTime}ms
                                    </Text>
                                  )}
                                  
                                  {result.error && (
                                    <Alert status="error" size="sm" mt={2}>
                                      <AlertIcon />
                                      <Text fontSize="xs">{result.error}</Text>
                                    </Alert>
                                  )}
                                  
                                  {result.result && (
                                    <Button
                                      size="xs"
                                      variant="ghost"
                                      onClick={onOpen}
                                      mt={2}
                                    >
                                      View Details
                                    </Button>
                                  )}
                                </CardBody>
                              </Card>
                            ))
                          )}
                        </VStack>
                      </CardBody>
                    </Card>
                  </GridItem>
                </Grid>
              </TabPanel>

              {/* Generation Tab */}
              <TabPanel>
                <Grid templateColumns="1fr 1fr" gap={6}>
                  <GridItem>
                    <Card>
                      <CardHeader>
                        <Heading size="md">Content Generation</Heading>
                      </CardHeader>
                      <CardBody>
                        <VStack spacing={4} align="stretch">
                          <ContentGenerator onGenerate={generateContent} />
                        </VStack>
                      </CardBody>
                    </Card>
                  </GridItem>
                  
                  <GridItem>
                    <Card>
                      <CardHeader>
                        <Heading size="md">Generated Content</Heading>
                      </CardHeader>
                      <CardBody>
                        <Text color="gray.500" textAlign="center">
                          Generated content will appear here
                        </Text>
                      </CardBody>
                    </Card>
                  </GridItem>
                </Grid>
              </TabPanel>

              {/* Analytics Tab */}
              <TabPanel>
                <Grid templateColumns="repeat(4, 1fr)" gap={6} mb={6}>
                  <Card>
                    <CardBody>
                      <Stat>
                        <StatLabel>Total Requests</StatLabel>
                        <StatNumber>{systemStats?.totalRequests.toLocaleString()}</StatNumber>
                        <StatHelpText>
                          <FiActivity />
                          +12% from last week
                        </StatHelpText>
                      </Stat>
                    </CardBody>
                  </Card>
                  
                  <Card>
                    <CardBody>
                      <Stat>
                        <StatLabel>Success Rate</StatLabel>
                        <StatNumber>
                          {systemStats ? 
                            ((systemStats.successfulRequests / systemStats.totalRequests) * 100).toFixed(1)
                            : 0}%
                        </StatNumber>
                        <StatHelpText>
                          <FiCheckCircle />
                          +2.3% from last week
                        </StatHelpText>
                      </Stat>
                    </CardBody>
                  </Card>
                  
                  <Card>
                    <CardBody>
                      <Stat>
                        <StatLabel>Avg Response Time</StatLabel>
                        <StatNumber>{systemStats?.averageResponseTime}s</StatNumber>
                        <StatHelpText>
                          <FiClock />
                          -0.2s from last week
                        </StatHelpText>
                      </Stat>
                    </CardBody>
                  </Card>
                  
                  <Card>
                    <CardBody>
                      <Stat>
                        <StatLabel>Active Users</StatLabel>
                        <StatNumber>{systemStats?.activeUsers}</StatNumber>
                        <StatHelpText>
                          <FiUsers />
                          +8 from yesterday
                        </StatHelpText>
                      </Stat>
                    </CardBody>
                  </Card>
                </Grid>

                <Grid templateColumns="2fr 1fr" gap={6}>
                  <Card>
                    <CardHeader>
                      <Heading size="md">Performance Metrics</Heading>
                    </CardHeader>
                    <CardBody>
                      <Line
                        data={performanceChartData}
                        options={{
                          responsive: true,
                          scales: {
                            y: {
                              type: 'linear',
                              display: true,
                              position: 'left',
                            },
                            y1: {
                              type: 'linear',
                              display: true,
                              position: 'right',
                              grid: {
                                drawOnChartArea: false,
                              },
                            },
                          },
                        }}
                      />
                    </CardBody>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <Heading size="md">Model Usage</Heading>
                    </CardHeader>
                    <CardBody>
                      <Doughnut
                        data={modelUsageData}
                        options={{
                          responsive: true,
                          plugins: {
                            legend: {
                              position: 'bottom',
                            },
                          },
                        }}
                      />
                    </CardBody>
                  </Card>
                </Grid>
              </TabPanel>

              {/* Models Tab */}
              <TabPanel>
                <Card>
                  <CardHeader>
                    <HStack justify="space-between">
                      <Heading size="md">Model Management</Heading>
                      <Button leftIcon={<FiRefreshCw />} onClick={fetchModels}>
                        Refresh
                      </Button>
                    </HStack>
                  </CardHeader>
                  <CardBody>
                    <VStack spacing={3} align="stretch">
                      {models.map((model) => (
                        <Card key={model.name} variant="outline">
                          <CardBody>
                            <HStack justify="space-between">
                              <VStack align="start" spacing={1}>
                                <Text fontWeight="bold">{model.name}</Text>
                                <HStack>
                                  <Badge size="sm">{model.type}</Badge>
                                  <Badge
                                    size="sm"
                                    colorScheme={model.status === 'loaded' ? 'green' : 'gray'}
                                  >
                                    {model.status}
                                  </Badge>
                                </HStack>
                                <Text fontSize="sm" color="gray.600">
                                  Memory: {model.memoryUsage}GB | Speed: {model.inferenceSpeed}
                                </Text>
                              </VStack>
                              
                              <HStack>
                                {model.status === 'loaded' && (
                                  <Progress
                                    value={model.memoryUsage * 20}
                                    size="sm"
                                    colorScheme="blue"
                                    w="100px"
                                  />
                                )}
                                <Button
                                  size="sm"
                                  colorScheme={model.status === 'loaded' ? 'red' : 'green'}
                                  variant="outline"
                                >
                                  {model.status === 'loaded' ? 'Unload' : 'Load'}
                                </Button>
                              </HStack>
                            </HStack>
                          </CardBody>
                        </Card>
                      ))}
                    </VStack>
                  </CardBody>
                </Card>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Box>

        {/* Result Details Modal */}
        <Modal isOpen={isOpen} onClose={onClose} size="xl">
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Analysis Details</ModalHeader>
            <ModalCloseButton />
            <ModalBody pb={6}>
              <Text>Detailed analysis results would be displayed here...</Text>
            </ModalBody>
          </ModalContent>
        </Modal>
      </Box>
    </ChakraProvider>
  );
};

// Content Generator Component
const ContentGenerator: React.FC<{
  onGenerate: (prompt: string, contentType: string) => Promise<any>;
}> = ({ onGenerate }) => {
  const [prompt, setPrompt] = useState('');
  const [contentType, setContentType] = useState('text');
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    try {
      await onGenerate(prompt, contentType);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <VStack spacing={4} align="stretch">
      <Box>
        <Text mb={2} fontWeight="medium">Content Type</Text>
        <Select value={contentType} onChange={(e) => setContentType(e.target.value)}>
          <option value="text">Text</option>
          <option value="image">Image</option>
          <option value="creative_writing">Creative Writing</option>
          <option value="code">Code</option>
          <option value="poem">Poem</option>
          <option value="story">Story</option>
        </Select>
      </Box>
      
      <Box>
        <Text mb={2} fontWeight="medium">Prompt</Text>
        <Textarea
          placeholder="Enter your generation prompt..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
        />
      </Box>
      
      <Button
        colorScheme="purple"
        onClick={handleGenerate}
        isLoading={isGenerating}
        loadingText="Generating..."
        leftIcon={<FiImage />}
      >
        Generate Content
      </Button>
    </VStack>
  );
};

export default App;
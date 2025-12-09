#!/bin/bash
# =============================================================================
# RAG Finance System - Docker Quick Start Script
# =============================================================================
# This script helps you quickly set up and start the RAG Finance System
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed ($(docker --version))"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed ($(docker-compose --version))"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker daemon is running"
    
    echo ""
}

# Check environment variables
check_env() {
    print_header "Checking Environment Variables"
    
    if [ ! -f .env ]; then
        print_warning ".env file not found"
        read -p "Would you like to create one now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "Enter your OpenAI API key: " openai_key
            cat > .env << EOF
# RAG Finance System - Environment Variables
OPENAI_API_KEY=$openai_key
VECTOR_STORE_MODE=chroma
MAX_CORRECTIONS=2
LOG_LEVEL=INFO
EOF
            print_success ".env file created"
        else
            print_error "Cannot continue without .env file"
            exit 1
        fi
    else
        print_success ".env file exists"
    fi
    
    # Check if OPENAI_API_KEY is set
    if ! grep -q "OPENAI_API_KEY=sk-" .env; then
        print_warning "OPENAI_API_KEY may not be set correctly in .env"
        print_info "Make sure to set your OpenAI API key in the .env file"
    else
        print_success "OPENAI_API_KEY is configured"
    fi
    
    echo ""
}

# Build and start services
start_services() {
    print_header "Building and Starting Services"
    
    print_info "Building Docker images (this may take a few minutes)..."
    docker-compose build
    
    print_info "Starting services..."
    docker-compose up -d
    
    print_success "All services started successfully!"
    echo ""
}

# Wait for services to be healthy
wait_for_services() {
    print_header "Waiting for Services to be Ready"
    
    print_info "Waiting for RAG API to be healthy..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_success "RAG API is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "RAG API failed to start within 60 seconds"
            print_info "Check logs with: docker-compose logs rag-api"
            exit 1
        fi
        sleep 2
    done
    
    print_info "Checking other services..."
    sleep 3
    
    # Check Jaeger
    if curl -f http://localhost:16686 &> /dev/null; then
        print_success "Jaeger UI is ready!"
    else
        print_warning "Jaeger UI may not be ready yet"
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        print_success "Prometheus is ready!"
    else
        print_warning "Prometheus may not be ready yet"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        print_success "Grafana is ready!"
    else
        print_warning "Grafana may not be ready yet"
    fi
    
    echo ""
}

# Print access information
print_access_info() {
    print_header "Access Information"
    
    echo -e "${GREEN}Services are now running!${NC}"
    echo ""
    echo -e "${BLUE}📡 RAG Finance API:${NC}"
    echo -e "   URL:  http://localhost:8000"
    echo -e "   Docs: http://localhost:8000/docs"
    echo ""
    echo -e "${BLUE}🔍 Jaeger (Distributed Tracing):${NC}"
    echo -e "   URL:  http://localhost:16686"
    echo ""
    echo -e "${BLUE}📊 Prometheus (Metrics):${NC}"
    echo -e "   URL:  http://localhost:9090"
    echo ""
    echo -e "${BLUE}📈 Grafana (Dashboards):${NC}"
    echo -e "   URL:      http://localhost:3000"
    echo -e "   Username: admin"
    echo -e "   Password: admin"
    echo ""
    
    print_header "Quick Test"
    echo "Test the API with:"
    echo ""
    echo -e "${YELLOW}curl -X POST http://localhost:8000/query \\${NC}"
    echo -e "${YELLOW}  -H 'Content-Type: application/json' \\${NC}"
    echo -e "${YELLOW}  -d '{\"query\": \"What was the revenue in Q4 2024?\"}' | jq${NC}"
    echo ""
    
    print_header "Useful Commands"
    echo "View logs:           docker-compose logs -f"
    echo "View specific logs:  docker-compose logs -f rag-api"
    echo "Stop services:       docker-compose down"
    echo "Restart services:    docker-compose restart"
    echo "Check status:        docker-compose ps"
    echo ""
}

# Main execution
main() {
    clear
    print_header "RAG Finance System - Docker Quick Start"
    echo ""
    
    check_prerequisites
    check_env
    start_services
    wait_for_services
    print_access_info
    
    print_success "Setup complete! 🎉"
}

# Run main function
main


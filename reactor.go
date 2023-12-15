package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"
)

// Job 表示一个工作单元
type Job struct {
	Conn net.Conn
}

// Worker 表示一个工作协程
type Worker struct {
	WorkerPool chan chan Job
	JobChannel chan Job
	Quit       chan bool
}

// NewWorker 创建一个新的工作协程
func NewWorker(workerPool chan chan Job) Worker {
	return Worker{
		WorkerPool: workerPool,
		JobChannel: make(chan Job),
		Quit:       make(chan bool),
	}
}

// Start 让工作协程开始监听工作
func (w Worker) Start() {
	go func() {
		for {
			// 将当前的工作协程注册到工作池中
			w.WorkerPool <- w.JobChannel

			select {
			case job := <-w.JobChannel:
				// 接收到工作，处理连接
				handleConnection(job.Conn)
			case <-w.Quit:
				// 接收到退出信号
				return
			}
		}
	}()
}

// Stop 使工作协程停止监听工作
func (w Worker) Stop() {
	go func() {
		w.Quit <- true
	}()
}

// 负责处理连接的函数
func handleConnection(conn net.Conn) {
	defer conn.Close()
	reader := bufio.NewReader(conn)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Print("Received: ", string(line))
		conn.Write([]byte(line))
	}
}

// Dispatcher 负责将工作分发给工作协程
type Dispatcher struct {
	WorkerPool chan chan Job
	MaxWorkers int
}

// NewDispatcher 创建一个新的调度器
func NewDispatcher(maxWorkers int) *Dispatcher {
	pool := make(chan chan Job, maxWorkers)
	return &Dispatcher{WorkerPool: pool, MaxWorkers: maxWorkers}
}

// Run 启动调度器
func (d *Dispatcher) Run() {
	for i := 0; i < d.MaxWorkers; i++ {
		worker := NewWorker(d.WorkerPool)
		worker.Start()
	}

	go d.dispatch()
}

// dispatch 负责监听连接并将它们分配给工作协程
func (d *Dispatcher) dispatch() {
	ln, err := net.Listen("tcp", ":8080")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer ln.Close()

	for {
		conn, err := ln.Accept()
		if err != nil {
			fmt.Println(err)
			continue
		}

		job := Job{Conn: conn}
		jobChannel := <-d.WorkerPool
		jobChannel <- job
	}
}

func main() {
	dispatcher := NewDispatcher(5) // 创建一个含有 5 个工作协程的调度器
	dispatcher.Run()

	// 创建一个通道来监听系统信号
	signalChannel := make(chan os.Signal, 1)
	// 监听 SIGINT 和 SIGTERM
	signal.Notify(signalChannel, os.Interrupt, syscall.SIGTERM)

	// 阻塞，直到收到信号
	<-signalChannel

	// 收到信号后，程序退出
	println("Shutting down...") // 运行调度器
}

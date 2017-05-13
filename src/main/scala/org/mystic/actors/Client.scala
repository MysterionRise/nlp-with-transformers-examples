package org.mystic.actors

import java.net.InetSocketAddress

import akka.actor.{Actor, ActorLogging, Props}
import akka.io.{IO, Tcp}

class Client(hostName: String, port: Int) extends Actor with ActorLogging {

  import Tcp._
  import context.system

  // connecting to the server for the trade data
  IO(Tcp) ! Connect(new InetSocketAddress(hostName, port))

  override def receive: Receive = {

    case CommandFailed(_: Connect) =>
      log.info(s"couldn't connect to $hostName $port")
      Thread.sleep(60000)
      log.info(s"tryint to connect to $hostName $port once again")
      IO(Tcp) ! Connect(new InetSocketAddress(hostName, port))

    case Connected(remoteAddress, localAddress) =>

      val handler = context.actorOf(Props[ReadHandler])
      sender() ! Register(handler)

    case _ =>
      log.info("something goes wrong in client")
  }
}

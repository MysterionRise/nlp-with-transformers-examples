package org.mystic.actors

import java.net.InetSocketAddress
import java.time.LocalDateTime

import akka.actor.{Actor, ActorLogging, Props}
import akka.io.{IO, Tcp}
import org.mystic.model.{Cancel, Last10Minutes, Last1Minute}

import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

class Server(hostName: String, port: Int) extends Actor with ActorLogging {

  import Tcp._
  import context.system

  // accepting connections from the clients on hostname port
  IO(Tcp) ! Bind(self, new InetSocketAddress(hostName, port))

  override def receive: Receive = {

    case Bound(localAddress) =>
      log.info(s"accepting the connection on $localAddress")

    case CommandFailed(_: Bind) => context stop self

    case Connected(remoteAddress, localAddress) =>

      log.info(s"somebody connected from $remoteAddress")

      val handler = context.actorOf(Props(classOf[WriteHandler], sender()))

      // we need to send data for last 10 minutes
      sender() ! Register(handler)
      handler ! Last10Minutes
      val cancelable = system.scheduler.schedule((60 - LocalDateTime.now().getSecond) seconds, 1 minutes, handler, Last1Minute)
      handler ! Cancel(cancelable)

    case _ =>
      log.info("something goes wrong in server")
  }
}

package org.mystic.actors

import akka.actor.{Actor, ActorLogging, ActorRef, Cancellable}
import akka.io.Tcp.{PeerClosed, Write}
import akka.util.ByteString
import org.mystic.model._

class WriteHandler(whereToSend: ActorRef) extends Actor with ActorLogging {

  var cancelable: Cancellable = _

  override def receive: Receive = {

    case Last10Minutes => {
      log.info("trying to send 10 minutes data to the client")
      val storage = context.system.actorSelection("user/storage")
      storage ! AskFor10MData
    }

    case Data10Minutes(data: List[Option[CandleDeal]]) => {
      data.filter(_.isDefined).map(_.get).map(_.toJson()).foreach(x => {
        whereToSend ! Write(ByteString(x))
      })
    }

    case Data1Minutes(data: List[Option[CandleDeal]]) => {
      data.filter(_.isDefined).map(_.get).map(_.toJson()).foreach(x => {
        whereToSend ! Write(ByteString(x))
      })
    }

    case Cancel(cancel) => {
      log.info("setting the cancellable")
      cancelable = cancel
    }

    case Last1Minute => {
      log.info("trying to send last minute data to the client")
      val storage = context.system.actorSelection("user/storage")
      storage ! AskFor1MData
    }

    case PeerClosed => {
      log.info("peer closed")
      cancelable.cancel()
      context stop self
    }

    case _ => log.error("something goes wrong in WriteHandler")
  }

}

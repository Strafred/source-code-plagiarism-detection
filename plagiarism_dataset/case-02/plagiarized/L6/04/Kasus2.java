import java.util.Scanner;
public class Kasus2 {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        double r = input.nextDouble();
        double l = input.nextDouble();
        System.out.println("The area is " + r*r*Math.PI);
        System.out.println("The volume of the cylinder is " + r*r*Math.PI*l);
    }
}
